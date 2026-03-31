"""app/langfuse_view.py — LangFuse observability view model.

This is the **single view-model layer** between the LangFuse integration and
the Streamlit rendering code.  It never calls Streamlit or the LangFuse SDK
directly — it reads from the pipeline result dict and returns structured
Python data-classes that ``app/main.py`` renders.

Architecture
------------
::

    PipelineState result dict
          │
          ▼
    build_langfuse_run_summary()   ←  reads: langfuse_trace_id,
          │                              graph_trace, memory_trace,
          │                              rag_* fields, settings
          ▼
    LangfuseRunSummary (frozen dataclass)
    ├── node_summaries: list[LangfuseNodeSummary]
    └── all other fields
          │
          ▼
    app/main.py  →  _render_observability_tab()

Design goals
------------
* **Pure Python** — no Streamlit, no LangFuse SDK calls.
* **Graceful degradation** — every field has a safe default; parsing
  failures never propagate.
* **Reusable** — future agents (Text-to-Code, Critic …) appear
  automatically because node summaries are derived from ``graph_trace``.
* **Stable contract** — the dataclasses are the UI contract; callers
  depend on fields, not on raw dict keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config.settings import settings
from utils.langfuse_client import is_langfuse_enabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-classes returned to the UI layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LangfuseNodeSummary:
    """Per-node LangFuse observability summary."""

    node_name: str
    status: str  # "success" | "failed" | "skipped" | "unknown"
    duration_s: float | None
    model_name: str | None  # drawn from NodeTelemetry.model_info
    provider: str | None  # "ollama" | "gemini" | None
    summary: str  # human-readable outcome from graph_trace
    error: str | None  # error message if status == "failed"


@dataclass(frozen=True)
class LangfuseRunSummary:
    """Full observability summary for one pipeline run.

    Constructed by :func:`build_langfuse_run_summary` from the completed
    pipeline state dict.  Consumed exclusively by the Streamlit rendering
    layer in ``app/main.py``.
    """

    # -- LangFuse connectivity --
    enabled: bool
    host: str | None
    trace_id: str | None
    trace_url: str | None  # pre-built URL into LangFuse UI, or None

    # -- Run identity --
    dataset_id: str | None
    file_name: str | None

    # -- Overall outcome --
    status: str  # "success" | "partial" | "failed" | "unknown" | "disabled"
    started_at: str | None  # ISO-8601 string from first node trace
    duration_s: float | None  # wall-clock pipeline duration

    # -- Node breakdown --
    node_summaries: list[LangfuseNodeSummary] = field(default_factory=list)
    nodes_succeeded: int = 0
    nodes_failed: int = 0
    nodes_skipped: int = 0

    # -- RAG / memory --
    memory_events_count: int = 0
    memory_stored: bool = False
    memory_collections: list[str] = field(default_factory=list)
    rag_context_found: bool = False
    memory_events_summary: str | None = None


# ---------------------------------------------------------------------------
# URL builder helper
# ---------------------------------------------------------------------------


def build_trace_url(host: str | None, trace_id: str | None) -> str | None:
    """Build a LangFuse trace URL from host and trace id.

    Returns ``None`` if either argument is absent or empty.

    The LangFuse v2 URL format is::

        {host}/trace/{trace_id}

    This function is the *single place* where this URL pattern is encoded.
    If LangFuse changes its URL structure, edit this function only.
    """
    if not host or not trace_id:
        return None
    return f"{host.rstrip('/')}/trace/{trace_id}"


# ---------------------------------------------------------------------------
# Node summary builder
# ---------------------------------------------------------------------------


def _extract_node_summary(entry: dict[str, Any]) -> LangfuseNodeSummary:
    """Build a :class:`LangfuseNodeSummary` from a ``NodeTraceEntry`` dict."""
    telem: dict[str, Any] = entry.get("telemetry") or {}
    model_info: dict[str, Any] = telem.get("model_info") or {}
    return LangfuseNodeSummary(
        node_name=entry.get("node", "unknown"),
        status=entry.get("status", "unknown"),
        duration_s=entry.get("duration_s"),
        model_name=model_info.get("model_name"),
        provider=model_info.get("provider"),
        summary=entry.get("summary", ""),
        error=entry.get("error"),
    )


# ---------------------------------------------------------------------------
# Main view-model builder
# ---------------------------------------------------------------------------


def build_langfuse_run_summary(result: dict[str, Any]) -> LangfuseRunSummary:
    """Derive a :class:`LangfuseRunSummary` from the completed pipeline state.

    **Pure function** — no side effects; reads only from ``result`` and the
    module-level config/client imports.
    """
    enabled = is_langfuse_enabled()
    host = settings.LANGFUSE_BASE_URL if enabled else None
    trace_id = result.get("langfuse_trace_id") or None
    trace_url = build_trace_url(host, trace_id)

    dataset_id = result.get("dataset_id") or None
    file_name = result.get("file_name") or None

    # ── Node summaries from graph_trace ─────────────────────────────────
    graph_trace: list[dict[str, Any]] = result.get("graph_trace") or []

    # Deduplicate by node name — keep the last entry for each node
    # (LangGraph may emit intermediate partial updates for the same node)
    seen: dict[str, dict[str, Any]] = {}
    for entry in graph_trace:
        seen[entry.get("node", "?")] = entry

    node_summaries = [_extract_node_summary(e) for e in seen.values()]

    succeeded = sum(1 for n in node_summaries if n.status == "success")
    failed = sum(1 for n in node_summaries if n.status == "failed")
    skipped = sum(1 for n in node_summaries if n.status == "skipped")

    # Derive overall status
    if not enabled:
        overall_status = "disabled"
    elif not node_summaries:
        overall_status = "unknown"
    elif failed > 0 and succeeded == 0:
        overall_status = "failed"
    elif failed > 0:
        overall_status = "partial"
    elif succeeded > 0:
        overall_status = "success"
    else:
        overall_status = "unknown"

    # ── Timing ──────────────────────────────────────────────────────────
    started_at: str | None = None
    duration_s: float | None = None
    if graph_trace:
        started_at = graph_trace[0].get("started_at")
        try:
            from datetime import datetime

            t0 = datetime.fromisoformat(graph_trace[0]["started_at"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(graph_trace[-1]["finished_at"].replace("Z", "+00:00"))
            duration_s = round((t1 - t0).total_seconds(), 2)
        except Exception:
            # Fall back to summing per-node durations
            totals = [
                e.get("duration_s", 0)
                for e in graph_trace
                if isinstance(e.get("duration_s"), int | float)
            ]
            duration_s = round(sum(totals), 2) if totals else None

    # ── Memory / RAG ─────────────────────────────────────────────────────
    memory_trace: list[dict[str, Any]] = result.get("memory_trace") or []
    collections: list[str] = []
    for evt in memory_trace:
        col = evt.get("collection")
        if col and col not in collections:
            collections.append(col)

    return LangfuseRunSummary(
        enabled=enabled,
        host=host,
        trace_id=trace_id,
        trace_url=trace_url,
        dataset_id=dataset_id,
        file_name=file_name,
        status=overall_status,
        started_at=started_at,
        duration_s=duration_s,
        node_summaries=node_summaries,
        nodes_succeeded=succeeded,
        nodes_failed=failed,
        nodes_skipped=skipped,
        memory_events_count=len(memory_trace),
        memory_stored=bool(result.get("rag_stored")),
        memory_collections=collections,
        rag_context_found=bool(result.get("rag_analysis_context")),
        memory_events_summary=result.get("rag_summary") or None,
    )
