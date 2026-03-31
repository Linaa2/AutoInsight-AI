"""LangGraph orchestration graph for AutoInsight-AI.

This module defines the **canonical** multi-agent workflow as a
``StateGraph``.  Every agent is wired as a thin node function that:

1. Reads inputs from :class:`~orchestration.state.PipelineState`.
2. Delegates to the corresponding agent class in ``agents/``.
3. Writes outputs (including a trace entry) back into the state.

Graph flow::

    START → profiler_node → analyst_node → visualizer_node → reporter_node → rag_storage_node → END
    START → profiler_node → analyst_node → critic_node → uncertainty_node → visualizer_node → reporter_node → rag_storage_node → END

Usage::

    from orchestration.graph import build_graph, run_analysis

    result = run_analysis(df, file_name="sales.csv")
"""

from __future__ import annotations

import dataclasses
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.profiler import ProfilerAgent
from agents.reporter import ReporterAgent
from agents.uncertainty import UncertaintyEstimator
from agents.visualizer import run_visualization_pipeline
from diagnostics.telemetry import (
    NodeTelemetry,
    build_telemetry,
    build_telemetry_no_llm,
    collect_resource_snapshot,
)
from orchestration.state import MemoryTraceEntry, NodeTraceEntry, PipelineState
from tools.profiler_engine import DataProfiler
from utils.langfuse_client import monitor as lf_monitor
from visualization.schemas import VisualizerRequest

logger = logging.getLogger(__name__)

_DF_REGISTRY: dict[str, pd.DataFrame] = {}
_AGENTS_ORDER = ["profiler", "analyst", "critic", "uncertainty", "visualizer", "reporter"]


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _trace_entry(
    node: str,
    *,
    status: str,
    started: float,
    keys_read: list[str],
    keys_written: list[str],
    summary: str = "",
    error: str | None = None,
    telemetry: NodeTelemetry | None = None,
) -> NodeTraceEntry:
    """Build a :class:`NodeTraceEntry` for a node execution."""
    finished = time.time()
    entry = NodeTraceEntry(
        node=node,
        status=status,
        started_at=datetime.fromtimestamp(started, tz=UTC).isoformat(),
        finished_at=datetime.fromtimestamp(finished, tz=UTC).isoformat(),
        duration_s=round(finished - started, 3),
        keys_read=keys_read,
        keys_written=keys_written,
        summary=summary,
        error=error,
    )
    if telemetry:
        entry["telemetry"] = cast("dict[str, Any]", telemetry)
    return entry


def _append_trace(state: PipelineState, entry: NodeTraceEntry) -> list[NodeTraceEntry]:
    """Return the existing trace list with *entry* appended."""
    existing: list[NodeTraceEntry] = list(state.get("graph_trace") or [])
    existing.append(entry)
    return existing


def _append_memory_trace(
    state: PipelineState,
    entries: list[MemoryTraceEntry],
) -> list[MemoryTraceEntry]:
    """Return the existing memory trace list with *entries* appended."""
    existing: list[MemoryTraceEntry] = list(state.get("memory_trace") or [])
    existing.extend(entries)
    return existing


def _dataset_key(state: PipelineState) -> str:
    """Return the pipeline-state key used to access the dataset."""
    return "df_ref" if state.get("df_ref") else "df_dict"


def _register_dataframe(df: pd.DataFrame) -> str:
    """Store *df* in the in-process registry and return its reference key."""
    import uuid

    ref = f"df-{uuid.uuid4().hex}"
    _DF_REGISTRY[ref] = df
    return ref


def _unregister_dataframe(ref: str | None) -> None:
    """Remove a registered DataFrame reference when the run completes."""
    if ref:
        _DF_REGISTRY.pop(ref, None)


def _resolve_dataframe(state: PipelineState) -> pd.DataFrame:
    """Resolve the DataFrame from the fast in-process registry or the legacy fallback."""
    df_ref = state.get("df_ref")
    if df_ref:
        df = _DF_REGISTRY.get(df_ref)
        if df is not None:
            return df

    df_dict = state.get("df_dict")
    if df_dict is not None:
        return pd.DataFrame(df_dict)

    raise KeyError("PipelineState is missing both 'df_ref' and 'df_dict'.")


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def profiler_node(state: PipelineState) -> PipelineState:
    """Run deterministic profiling + LLM interpretation.

    Reads:  ``df_ref`` or ``df_dict``
    Writes: ``profile_data``, ``profile_markdown``
    """
    t0 = time.time()
    res_before = collect_resource_snapshot()
    callbacks = lf_monitor.get_llm_callbacks()
    node_meta = {"dataset_id": state.get("dataset_id", ""), "file_name": state.get("file_name", "")}
    with lf_monitor.node_span("profiler", metadata=node_meta):
        try:
            dataset_key = _dataset_key(state)
            df = _resolve_dataframe(state)
            profiler = DataProfiler()
            profile = profiler.profile(df)

            agent = ProfilerAgent()
            llm_start = time.time()
            markdown = agent.describe(profile, callbacks=callbacks)
            llm_end = time.time()

            res_after = collect_resource_snapshot()
            telem = build_telemetry(
                task="text",
                llm_start=llm_start,
                llm_end=llm_end,
                resource_before=res_before,
                resource_after=res_after,
            )
            entry = _trace_entry(
                "profiler",
                status="success",
                started=t0,
                keys_read=[dataset_key],
                keys_written=["profile_data", "profile_markdown"],
                summary=f"Profiled {profile.shape[0]} rows x {profile.shape[1]} cols",
                telemetry=telem,
            )
            return {
                "profile_data": profile.to_dict(),
                "profile_markdown": markdown,
                "graph_trace": _append_trace(state, entry),
            }
        except Exception as exc:
            msg = f"Profiler failed: {exc}"
            logger.exception(msg)
            res_after = collect_resource_snapshot()
            telem = build_telemetry_no_llm(resource_before=res_before, resource_after=res_after)
            entry = _trace_entry(
                "profiler",
                status="failed",
                started=t0,
                keys_read=[_dataset_key(state)],
                keys_written=[],
                error=msg,
                telemetry=telem,
            )
            return {"error": msg, "graph_trace": _append_trace(state, entry)}


def analyst_node(state: PipelineState) -> PipelineState:
    """Generate structured insights from the profile.

    Reads:  ``df_ref`` or ``df_dict``, ``profile_data``, ``profile_markdown``
    Writes: ``insights``, ``insights_markdown``
    """
    t0 = time.time()
    profile_md = state.get("profile_markdown")
    if not profile_md:
        entry = _trace_entry(
            "analyst",
            status="skipped",
            started=t0,
            keys_read=["profile_markdown"],
            keys_written=[],
            summary="Skipped — no profile_markdown available",
        )
        return {"graph_trace": _append_trace(state, entry)}

    callbacks = lf_monitor.get_llm_callbacks()
    node_meta = {"dataset_id": state.get("dataset_id", ""), "file_name": state.get("file_name", "")}
    res_before = collect_resource_snapshot()
    with lf_monitor.node_span("analyst", metadata=node_meta):
        try:
            dataset_key = _dataset_key(state)
            df = _resolve_dataframe(state)
            sample_text = df.head(5).to_string()
            profile_data = state.get("profile_data")

            agent = AnalystAgent()
            llm_start = time.time()
            result = agent.run(
                profiler_output=profile_md,
                sample_text=sample_text,
                profile_data=profile_data,
                callbacks=callbacks,
            )
            llm_end = time.time()

            insights = result.get("insights", [])
            insights_md = result.get("analyst_output", "")

            res_after = collect_resource_snapshot()
            telem = build_telemetry(
                task="text",
                llm_start=llm_start,
                llm_end=llm_end,
                resource_before=res_before,
                resource_after=res_after,
            )
            entry = _trace_entry(
                "analyst",
                status="success",
                started=t0,
                keys_read=[dataset_key, "profile_data", "profile_markdown"],
                keys_written=["insights", "insights_markdown"],
                summary=f"Generated {len(insights)} insights",
                telemetry=telem,
            )
            return {
                "insights": insights,
                "insights_markdown": insights_md,
                "graph_trace": _append_trace(state, entry),
            }
        except Exception as exc:
            msg = f"Analyst failed: {exc}"
            logger.exception(msg)
            res_after = collect_resource_snapshot()
            telem = build_telemetry_no_llm(resource_before=res_before, resource_after=res_after)
            entry = _trace_entry(
                "analyst",
                status="failed",
                started=t0,
                keys_read=[_dataset_key(state), "profile_data", "profile_markdown"],
                keys_written=[],
                error=msg,
                telemetry=telem,
            )
            return {"error": msg, "graph_trace": _append_trace(state, entry)}


def critic_node(state: PipelineState) -> PipelineState:
    """Run adversarial critique on each analyst insight.

    Reads:  ``insights``, ``profile_data``
    Writes: ``critiques``, ``critic_output``
    """
    t0 = time.time()
    insights: list[dict[str, Any]] = state.get("insights") or []

    if not insights:
        entry = _trace_entry(
            "critic",
            status="skipped",
            started=t0,
            keys_read=["insights"],
            keys_written=[],
            summary="Skipped — no insights to critique",
        )
        return {"graph_trace": _append_trace(state, entry)}

    profile_data: dict[str, Any] = state.get("profile_data") or {}
    node_meta = {"dataset_id": state.get("dataset_id", ""), "file_name": state.get("file_name", "")}
    res_before = collect_resource_snapshot()

    with lf_monitor.node_span("critic", metadata=node_meta):
        try:
            agent = CriticAgent()
            llm_start = time.time()
            result = agent.run(insights=insights, profile_data=profile_data)
            llm_end = time.time()

            critiques: list[dict[str, Any]] = result.get("critiques", [])
            critic_md: str = result.get("critic_output", "")

            res_after = collect_resource_snapshot()
            telem = build_telemetry(
                task="text",
                llm_start=llm_start,
                llm_end=llm_end,
                resource_before=res_before,
                resource_after=res_after,
            )
            entry = _trace_entry(
                "critic",
                status="success",
                started=t0,
                keys_read=["insights", "profile_data"],
                keys_written=["critiques", "critic_output"],
                summary=f"Critiqued {len(critiques)}/{len(insights)} insights",
                telemetry=telem,
            )
            return {
                "critiques": critiques,
                "critic_output": critic_md,
                "graph_trace": _append_trace(state, entry),
            }
        except Exception as exc:
            msg = f"CriticAgent failed: {exc}"
            logger.exception(msg)
            res_after = collect_resource_snapshot()
            telem = build_telemetry_no_llm(resource_before=res_before, resource_after=res_after)
            entry = _trace_entry(
                "critic",
                status="failed",
                started=t0,
                keys_read=["insights", "profile_data"],
                keys_written=[],
                error=msg,
                telemetry=telem,
            )
            return {"error": msg, "graph_trace": _append_trace(state, entry)}


def uncertainty_node(state: PipelineState) -> PipelineState:
    """Score each analyst insight with a 0-100% confidence estimate.

    Reads:  ``insights``, ``critiques`` (optional), ``profile_data``
    Writes: ``confidence_scores``, ``uncertainty_output``
    """
    t0 = time.time()
    insights: list[dict[str, Any]] = state.get("insights") or []

    if not insights:
        entry = _trace_entry(
            "uncertainty",
            status="skipped",
            started=t0,
            keys_read=["insights"],
            keys_written=[],
            summary="Skipped — no insights to score",
        )
        return {"graph_trace": _append_trace(state, entry)}

    critiques: list[dict[str, Any]] = state.get("critiques") or []
    profile_data: dict[str, Any] = state.get("profile_data") or {}
    callbacks = lf_monitor.get_llm_callbacks()
    node_meta = {"dataset_id": state.get("dataset_id", ""), "file_name": state.get("file_name", "")}
    res_before = collect_resource_snapshot()

    with lf_monitor.node_span("uncertainty", metadata=node_meta):
        try:
            estimator = UncertaintyEstimator()
            llm_start = time.time()
            result = estimator.estimate_all(insights, critiques, profile_data, callbacks)
            llm_end = time.time()

            scores: list[dict[str, Any]] = result["confidence_scores"]
            res_after = collect_resource_snapshot()
            telem = build_telemetry(
                task="text",
                llm_start=llm_start,
                llm_end=llm_end,
                resource_before=res_before,
                resource_after=res_after,
            )
            entry = _trace_entry(
                "uncertainty",
                status="success",
                started=t0,
                keys_read=["insights", "critiques", "profile_data"],
                keys_written=["confidence_scores", "uncertainty_output"],
                summary=f"Scored {len(scores)} insights",
                telemetry=telem,
            )
            return {
                "confidence_scores": scores,
                "uncertainty_output": result["uncertainty_output"],
                "graph_trace": _append_trace(state, entry),
            }
        except Exception as exc:
            msg = f"UncertaintyEstimator failed: {exc}"
            logger.exception(msg)
            res_after = collect_resource_snapshot()
            telem = build_telemetry_no_llm(resource_before=res_before, resource_after=res_after)
            entry = _trace_entry(
                "uncertainty",
                status="failed",
                started=t0,
                keys_read=["insights", "critiques", "profile_data"],
                keys_written=[],
                error=msg,
                telemetry=telem,
            )
            return {"error": msg, "graph_trace": _append_trace(state, entry)}


def visualizer_node(state: PipelineState) -> PipelineState:
    """Propose and execute charts based on the profile and insights.

    Reads:  ``df_ref`` or ``df_dict``, ``profile_markdown``, ``insights_markdown``
    Writes: ``visualization_result``
    """
    t0 = time.time()
    profile_md = state.get("profile_markdown")
    if not profile_md:
        entry = _trace_entry(
            "visualizer",
            status="skipped",
            started=t0,
            keys_read=["profile_markdown"],
            keys_written=[],
            summary="Skipped — no profile_markdown available",
        )
        return {"graph_trace": _append_trace(state, entry)}

    insights_md = state.get("insights_markdown") or profile_md
    callbacks = lf_monitor.get_llm_callbacks()
    node_meta = {"dataset_id": state.get("dataset_id", ""), "file_name": state.get("file_name", "")}
    res_before = collect_resource_snapshot()
    with lf_monitor.node_span("visualizer", metadata=node_meta):
        try:
            dataset_key = _dataset_key(state)
            df = _resolve_dataframe(state)
            columns_info = ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())
            request = VisualizerRequest(
                profile_markdown=profile_md,
                insights_markdown=insights_md,
                columns_info=columns_info,
            )

            llm_start = time.time()
            result = run_visualization_pipeline(df, request, callbacks=callbacks)
            llm_end = time.time()

            serialised: dict[str, Any] = {
                "raw_llm_output": result.raw_llm_output,
                "parsing_error": result.parsing_error,
                "charts": [
                    {
                        "spec": dataclasses.asdict(rc.spec),
                        "execution": {
                            "success": rc.execution.success,
                            "error": rc.execution.error,
                            "figure_json": (
                                rc.execution.figure.to_json()
                                if rc.execution.success and rc.execution.figure is not None
                                else None
                            ),
                        },
                    }
                    for rc in result.charts
                ],
            }

            n_ok = sum(1 for rc in result.charts if rc.execution.success)
            res_after = collect_resource_snapshot()
            telem = build_telemetry(
                task="code",
                llm_start=llm_start,
                llm_end=llm_end,
                resource_before=res_before,
                resource_after=res_after,
            )
            entry = _trace_entry(
                "visualizer",
                status="success",
                started=t0,
                keys_read=[dataset_key, "profile_markdown", "insights_markdown"],
                keys_written=["visualization_result"],
                summary=f"{n_ok}/{len(result.charts)} charts rendered successfully",
                telemetry=telem,
            )
            return {
                "visualization_result": serialised,
                "graph_trace": _append_trace(state, entry),
            }
        except Exception as exc:
            msg = f"Visualizer failed: {exc}"
            logger.exception(msg)
            res_after = collect_resource_snapshot()
            telem = build_telemetry_no_llm(resource_before=res_before, resource_after=res_after)
            entry = _trace_entry(
                "visualizer",
                status="failed",
                started=t0,
                keys_read=[_dataset_key(state), "profile_markdown", "insights_markdown"],
                keys_written=[],
                error=msg,
                telemetry=telem,
            )
            return {"error": msg, "graph_trace": _append_trace(state, entry)}


def reporter_node(state: PipelineState) -> PipelineState:
    """Synthesize all prior outputs into an executive report.

    Reads:  ``profile_markdown``, ``insights``, ``insights_markdown``, ``visualization_result``
    Writes: ``report_markdown``
    """
    t0 = time.time()
    profile_md = state.get("profile_markdown")
    insights_md = state.get("insights_markdown")

    if not profile_md and not insights_md:
        entry = _trace_entry(
            "reporter",
            status="skipped",
            started=t0,
            keys_read=["profile_markdown", "insights_markdown"],
            keys_written=[],
            summary="Skipped — no profile or insights available",
        )
        return {"graph_trace": _append_trace(state, entry)}

    res_before = collect_resource_snapshot()
    callbacks = lf_monitor.get_llm_callbacks()
    node_meta = {"dataset_id": state.get("dataset_id", ""), "file_name": state.get("file_name", "")}
    with lf_monitor.node_span("reporter", metadata=node_meta):
        try:
            insights = state.get("insights")
            viz_result = state.get("visualization_result")
            dataset_id = state.get("dataset_id", "default")
            rag_context = state.get("rag_analysis_context") or ""
            memory_entries: list[MemoryTraceEntry] = []

            if not rag_context and dataset_id:
                try:
                    from agents.rag import RAGAgent

                    rag = RAGAgent(dataset_id=dataset_id)
                    rag_context = rag.get_analysis_context() or ""
                    memory_entries.append(
                        MemoryTraceEntry(
                            event="retrieve_context",
                            status="success" if rag_context else "empty",
                            dataset_id=dataset_id,
                            collection=None,
                            chunks=0,
                            message=(
                                f"Retrieved prior context ({len(rag_context)} chars)"
                                if rag_context
                                else "No prior context found"
                            ),
                        )
                    )
                except Exception as exc:
                    logger.debug("RAG context retrieval failed: %s", exc)
                    memory_entries.append(
                        MemoryTraceEntry(
                            event="retrieve_context",
                            status="failed",
                            dataset_id=dataset_id,
                            collection=None,
                            chunks=0,
                            message=str(exc),
                        )
                    )

                lf_monitor.log_event(
                    "rag-context-retrieval",
                    input={"dataset_id": dataset_id},
                    output={
                        "retrieved": bool(rag_context),
                        "context_chars": len(rag_context),
                    },
                )

            agent = ReporterAgent()
            llm_start = time.time()
            result = agent.run(
                profiler_output=profile_md or "",
                analyst_output=insights_md or "",
                insights=insights,
                visualizer_output=viz_result,
                rag_context=rag_context,
                critic_output=state.get("critic_output") or "",
                uncertainty_output=state.get("uncertainty_output") or "",
                callbacks=callbacks,
            )
            llm_end = time.time()

            report = result.get("reporter_output", "")
            memory_trace = _append_memory_trace(state, memory_entries) if memory_entries else None
            keys_written = ["report_markdown"]
            response: PipelineState = {
                "report_markdown": report,
            }
            if rag_context != (state.get("rag_analysis_context") or ""):
                response["rag_analysis_context"] = rag_context
                keys_written.append("rag_analysis_context")
            if memory_trace is not None:
                response["memory_trace"] = memory_trace
                keys_written.append("memory_trace")

            res_after = collect_resource_snapshot()
            telem = build_telemetry(
                task="text",
                llm_start=llm_start,
                llm_end=llm_end,
                resource_before=res_before,
                resource_after=res_after,
            )
            entry = _trace_entry(
                "reporter",
                status="success",
                started=t0,
                keys_read=[
                    "profile_markdown",
                    "insights",
                    "insights_markdown",
                    "visualization_result",
                    "rag_analysis_context",
                    "critic_output",
                    "uncertainty_output",
                ],
                keys_written=keys_written,
                summary=f"Report generated ({len(report)} chars)",
                telemetry=telem,
            )
            response["graph_trace"] = _append_trace(state, entry)
            return response
        except Exception as exc:
            msg = f"Reporter failed: {exc}"
            logger.exception(msg)
            res_after = collect_resource_snapshot()
            telem = build_telemetry_no_llm(resource_before=res_before, resource_after=res_after)
            entry = _trace_entry(
                "reporter",
                status="failed",
                started=t0,
                keys_read=[
                    "profile_markdown",
                    "insights",
                    "insights_markdown",
                    "visualization_result",
                ],
                keys_written=[],
                error=msg,
                telemetry=telem,
            )
            return {"error": msg, "graph_trace": _append_trace(state, entry)}


# ---------------------------------------------------------------------------
# RAG storage node
# ---------------------------------------------------------------------------


def rag_storage_node(state: PipelineState) -> PipelineState:
    """Persist analysis outputs to ChromaDB after the main pipeline completes.

    This is a *best-effort* node — it must never crash the analysis flow.
    Failures are recorded in ``memory_trace`` and ``graph_trace`` but do NOT
    propagate to the pipeline's ``error`` field.

    Reads:  ``dataset_id``, ``profile_markdown``, ``insights``,
            ``insights_markdown``, ``report_markdown``
    Writes: ``rag_stored``, ``rag_summary``, ``memory_trace``
    """
    t0 = time.time()
    res_before = collect_resource_snapshot()
    dataset_id = state.get("dataset_id", "default")

    profile_md = state.get("profile_markdown", "")
    insights: list[dict[str, Any]] = state.get("insights") or []
    insights_md = state.get("insights_markdown", "")
    report_md = state.get("report_markdown", "")

    # Nothing to store
    if not any([profile_md, insights, insights_md, report_md]):
        entry = _trace_entry(
            "rag_storage",
            status="skipped",
            started=t0,
            keys_read=["profile_markdown", "insights", "insights_markdown", "report_markdown"],
            keys_written=[],
            summary="Skipped — no analysis outputs to store",
        )
        return {
            "rag_stored": False,
            "rag_summary": "Nothing to store — pipeline produced no outputs.",
            "memory_trace": list(state.get("memory_trace") or []),
            "graph_trace": _append_trace(state, entry),
        }

    memory_trace: list[MemoryTraceEntry] = []
    stored_artifacts: list[str] = []
    total_chunks = 0

    try:
        from agents.rag import RAGAgent

        rag = RAGAgent(dataset_id=dataset_id)

        # Erase the previous run's chunks for this dataset before writing fresh
        # ones.  This prevents unbounded accumulation: without this call, every
        # analysis appends new vectors to ChromaDB and each subsequent run
        # retrieves an ever-growing mix of stale and current data.
        # The reporter_node already consumed the previous context (retrieval
        # happens before storage in the graph), so enrichment is unaffected.
        rag.store.clear_dataset(dataset_id)

        if profile_md:
            try:
                count = rag.store.store_profile(profile_md, dataset_id=dataset_id)
                total_chunks += count
                memory_trace.append(
                    MemoryTraceEntry(
                        event="store_profile",
                        status="success",
                        dataset_id=dataset_id,
                        collection="profiles",
                        chunks=count,
                        message="Profile markdown stored",
                    )
                )
                stored_artifacts.append("profile")
            except Exception as exc:
                logger.warning(f"RAG: failed to store profile: {exc}")
                memory_trace.append(
                    MemoryTraceEntry(
                        event="store_profile",
                        status="failed",
                        dataset_id=dataset_id,
                        collection="profiles",
                        chunks=0,
                        message=str(exc),
                    )
                )

        insights_payload: list[dict[str, Any]] | str = insights if insights else insights_md
        if insights_payload:
            try:
                count = rag.store.store_insights(insights_payload, dataset_id=dataset_id)
                total_chunks += count
                memory_trace.append(
                    MemoryTraceEntry(
                        event="store_insights",
                        status="success",
                        dataset_id=dataset_id,
                        collection="insights",
                        chunks=count,
                        message=f"Stored {len(insights)} structured insights"
                        if insights
                        else "Stored insights markdown",
                    )
                )
                stored_artifacts.append("insights")
            except Exception as exc:
                logger.warning(f"RAG: failed to store insights: {exc}")
                memory_trace.append(
                    MemoryTraceEntry(
                        event="store_insights",
                        status="failed",
                        dataset_id=dataset_id,
                        collection="insights",
                        chunks=0,
                        message=str(exc),
                    )
                )

        if report_md:
            try:
                count = rag.store.store_report(report_md, dataset_id=dataset_id)
                total_chunks += count
                memory_trace.append(
                    MemoryTraceEntry(
                        event="store_report",
                        status="success",
                        dataset_id=dataset_id,
                        collection="reports",
                        chunks=count,
                        message="Report markdown stored",
                    )
                )
                stored_artifacts.append("report")
            except Exception as exc:
                logger.warning(f"RAG: failed to store report: {exc}")
                memory_trace.append(
                    MemoryTraceEntry(
                        event="store_report",
                        status="failed",
                        dataset_id=dataset_id,
                        collection="reports",
                        chunks=0,
                        message=str(exc),
                    )
                )

        rag_stored = bool(stored_artifacts)
        rag_summary = (
            f"Stored {', '.join(stored_artifacts)} ({total_chunks} chunks) "
            f"for dataset '{dataset_id}'"
            if rag_stored
            else f"Nothing stored for dataset '{dataset_id}'"
        )

        # Emit a LangFuse event summarising the memory operation
        lf_monitor.log_event(
            "rag-storage",
            input={"dataset_id": dataset_id, "artifacts": stored_artifacts},
            output={"stored": rag_stored, "total_chunks": total_chunks, "summary": rag_summary},
        )

        res_after = collect_resource_snapshot()
        telem = build_telemetry_no_llm(resource_before=res_before, resource_after=res_after)
        entry = _trace_entry(
            "rag_storage",
            status="success" if rag_stored else "skipped",
            started=t0,
            keys_read=["profile_markdown", "insights", "insights_markdown", "report_markdown"],
            keys_written=["rag_stored", "rag_summary", "memory_trace"],
            summary=rag_summary,
            telemetry=telem,
        )
        return {
            "rag_stored": rag_stored,
            "rag_summary": rag_summary,
            "memory_trace": _append_memory_trace(state, memory_trace),
            "graph_trace": _append_trace(state, entry),
        }

    except Exception as exc:
        msg = f"RAG storage failed: {exc}"
        logger.warning(msg)
        res_after = collect_resource_snapshot()
        telem = build_telemetry_no_llm(resource_before=res_before, resource_after=res_after)
        entry = _trace_entry(
            "rag_storage",
            status="failed",
            started=t0,
            keys_read=["profile_markdown", "insights", "insights_markdown", "report_markdown"],
            keys_written=[],
            summary=msg,
            telemetry=telem,
        )
        return {
            "rag_stored": False,
            "rag_summary": msg,
            "memory_trace": _append_memory_trace(state, memory_trace),
            "graph_trace": _append_trace(state, entry),
        }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _get_node_action(node_name: str):
    """Helper to get the node function for a given node name."""
    try:
        match node_name:
            case "profiler":
                return profiler_node
            case "analyst":
                return analyst_node
            case "critic":
                return critic_node
            case "uncertainty":
                return uncertainty_node
            case "visualizer":
                return visualizer_node
            case "reporter":
                return reporter_node
            case "rag_storage":
                return rag_storage_node
            case _:
                raise ValueError(f"Unknown node name: {node_name}")
    except Exception as exc:
        msg = f"Error getting node action for {node_name}: {exc}"
        logger.exception(msg)
        return {"error": msg, "graph_trace": []}


def build_graph():
    """Build and compile the AutoInsight-AI orchestration graph.

    Graph flow::

        START → profiler → analyst → critic → uncertainty → visualizer → reporter → rag_storage → END

    Returns a compiled LangGraph ``CompiledGraph`` ready for ``.invoke()``.
    """
    graph = StateGraph(PipelineState)

    # Main analysis nodes
    for node_name in _AGENTS_ORDER:
        graph.add_node(node_name, _get_node_action(node_name))

    # RAG storage node — runs after reporter, best-effort
    graph.add_node("rag_storage", rag_storage_node)

    # Wire edges: START → profiler → analyst → visualizer → reporter → rag_storage → END
    graph.add_edge(START, _AGENTS_ORDER[0])
    for i in range(len(_AGENTS_ORDER) - 1):
        graph.add_edge(_AGENTS_ORDER[i], _AGENTS_ORDER[i + 1])
    graph.add_edge(_AGENTS_ORDER[-1], "rag_storage")
    graph.add_edge("rag_storage", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------


def _make_initial_state(
    df: pd.DataFrame,
    file_name: str,
    dataset_id: str | None = None,
) -> PipelineState:
    """Build the initial :class:`PipelineState` for a graph invocation.

    Computes ``dataset_id`` from the file name if not provided and registers
    the DataFrame in the in-process registry so nodes can access it by
    lightweight reference instead of rebuilding it from JSON on every step.
    Also starts a LangFuse trace for the run.
    """
    import uuid

    from utils.memory import ContextStore

    if not dataset_id:
        dataset_id = ContextStore.make_dataset_id(file_name)

    run_id = str(uuid.uuid4())[:8]

    # Best-effort: start a LangFuse trace for this analysis run
    langfuse_trace_id = lf_monitor.begin_run(
        dataset_id=dataset_id,
        file_name=file_name,
        run_id=run_id,
    )

    return PipelineState(
        df_ref=_register_dataframe(df),
        file_name=file_name,
        dataset_id=dataset_id,
        rag_analysis_context="",
        langfuse_trace_id=langfuse_trace_id,
        memory_trace=[],
        graph_trace=[],
    )


def run_analysis(
    df: pd.DataFrame,
    file_name: str = "dataset",
    dataset_id: str | None = None,
) -> PipelineState:
    """Run the full analysis pipeline and return the final state.

    This is the application-level entry point used by the Streamlit app
    and any other caller that wants the complete orchestrated result.

    Args:
        df:         The uploaded dataset as a pandas DataFrame.
        file_name:  Original file name (for display / tracing).
        dataset_id: Stable identifier for the dataset (auto-derived if omitted).

    Returns:
        The terminal :class:`PipelineState` with all agent outputs and the
        graph trace populated.
    """
    graph = build_graph()
    initial_state = _make_initial_state(df, file_name, dataset_id)
    df_ref = initial_state.get("df_ref")
    try:
        result: PipelineState = graph.invoke(initial_state)
        lf_monitor.end_run(
            status="success",
            output={"nodes_run": len(result.get("graph_trace") or [])},
        )
        result.pop("df_ref", None)
        return result
    except Exception:
        lf_monitor.end_run(status="error")
        raise
    finally:
        _unregister_dataframe(df_ref)
        lf_monitor.flush()


def stream_analysis(
    df: pd.DataFrame,
    file_name: str = "dataset",
    dataset_id: str | None = None,
) -> Iterator[tuple[str, PipelineState]]:
    """Stream the analysis pipeline, yielding after each node completes.

    Yields:
        ``(node_name, cumulative_state)`` tuples — one per completed node.
        The UI can render partial results as soon as each node finishes.

    Args:
        df:         The uploaded dataset as a pandas DataFrame.
        file_name:  Original file name (for display / tracing).
        dataset_id: Stable identifier for the dataset (auto-derived if omitted).
    """
    graph = build_graph()
    initial_state = _make_initial_state(df, file_name, dataset_id)
    df_ref = initial_state.get("df_ref")
    # Build a plain dict of stable fields so node_output merges stay type-clean.
    # We access via .get() with a type: ignore to avoid the TypedDict literal-key
    # restriction — the keys are all valid PipelineState fields.
    static_state: dict[str, Any] = {}
    for _skey in ("file_name", "dataset_id", "langfuse_trace_id"):
        _sval = initial_state.get(_skey)  # type: ignore[misc]
        if _sval is not None:
            static_state[_skey] = _sval
    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                # stream_mode="updates" only yields node deltas; inject the stable
                # run metadata so the progressive UI sees the same identity fields
                # as the non-streaming ``run_analysis`` path.
                node_output = {**static_state, **node_output}
                node_output.pop("df_ref", None)
                yield node_name, cast("PipelineState", node_output)
        lf_monitor.end_run(status="success")
    except Exception:
        lf_monitor.end_run(status="error")
        raise
    finally:
        _unregister_dataframe(df_ref)
        lf_monitor.flush()
