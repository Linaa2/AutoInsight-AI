"""Shared graph state for the LangGraph orchestration pipeline.

The :class:`PipelineState` ``TypedDict`` is the **single authoritative
contract** between all graph nodes.  Each node reads what it needs, writes
its outputs, and leaves everything else untouched.

Design choices
--------------
* **TypedDict** with ``total=False`` — LangGraph's ``StateGraph`` natively
  supports TypedDicts for key-based state updates.
* ``df_dict`` (``list[dict]``) is used instead of a raw ``DataFrame`` because
  LangGraph serialises state when checkpointing; a dict-list is JSON-safe.
  Nodes reconstruct the ``DataFrame`` locally.
* A ``graph_trace`` list accumulates one entry per executed node, giving the
  UI layer full observability over the pipeline run.
"""

from __future__ import annotations

from typing import Any, TypedDict


class NodeTraceEntry(TypedDict, total=False):
    """Diagnostic record for a single node execution."""

    node: str
    status: str  # "success" | "skipped" | "failed"
    started_at: str  # ISO-8601
    finished_at: str  # ISO-8601
    duration_s: float
    keys_read: list[str]
    keys_written: list[str]
    summary: str
    error: str | None
    telemetry: dict[str, Any]  # NodeTelemetry from diagnostics.telemetry


class MemoryTraceEntry(TypedDict, total=False):
    """Record of a single ChromaDB memory interaction."""

    event: str  # "store_profile" | "store_insights" | "store_report" | "retrieve_context"
    status: str  # "success" | "skipped" | "empty" | "failed"
    dataset_id: str
    collection: str | None
    chunks: int  # number of chunks stored / retrieved
    message: str


class PipelineState(TypedDict, total=False):
    """Shared state flowing through the LangGraph pipeline.

    All fields are optional (``total=False``) so each node only returns
    the keys it produces, without needing to echo back the rest.
    """

    # ---- inputs (set before graph invocation) ----
    df_dict: list[dict[str, Any]]
    file_name: str
    dataset_id: str  # stable identifier derived from file_name via ContextStore.make_dataset_id

    # ---- profiler outputs ----
    profile_data: dict[str, Any]
    profile_markdown: str

    # ---- analyst outputs ----
    insights: list[dict[str, Any]]
    insights_markdown: str

    # ---- visualizer outputs ----
    visualization_result: dict[str, Any]

    # ---- reporter outputs ----
    report_markdown: str

    # ---- RAG / memory fields ----
    rag_analysis_context: str  # retrieved context injected into reporter (from previous runs)
    rag_stored: bool  # True when rag_storage_node ran successfully
    rag_summary: str  # human-readable summary of what was stored
    memory_trace: list[MemoryTraceEntry]  # one entry per memory interaction

    # ---- critic / self-correction (future) ----
    feedback: str

    # ---- diagnostics ----
    graph_trace: list[NodeTraceEntry]
    error: str
