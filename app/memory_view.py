"""app/memory_view.py — Memory / RAG presentation layer for the Streamlit UI.

This module is the **view-model** between the RAG back-end (``agents/rag.py``
and ``utils/memory.py``) and the Streamlit rendering code.  It never calls
Streamlit directly — it returns plain Python values that ``app/main.py``
renders.

Design goals:
    * Keep all ChromaDB / RAG interactions behind a single service facade.
    * Return structured data, not rendered HTML — the caller decides layout.
    * Graceful degradation — every function returns a safe default on failure.
    * Reusable by future agents (Text-to-Code, conversational Q&A).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes returned to the UI layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryStatus:
    """Summary of memory state for a single analysis run."""

    dataset_id: str = ""
    memory_available: bool = False
    previous_context_found: bool = False
    previous_context_chars: int = 0
    current_run_stored: bool = False
    rag_summary: str = ""
    memory_trace: list[dict[str, Any]] = field(default_factory=list)
    stored_artifacts: list[str] = field(default_factory=list)
    collections_used: list[str] = field(default_factory=list)
    total_chunks_stored: int = 0


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieval result from memory — ready for rendering."""

    label: str
    content: str
    empty: bool = True


# ---------------------------------------------------------------------------
# Build memory status from pipeline state
# ---------------------------------------------------------------------------


def build_memory_status(result: dict[str, Any]) -> MemoryStatus:
    """Derive a :class:`MemoryStatus` from the completed pipeline state.

    This is a **pure function** with no side effects — it only reads
    fields already present in the pipeline result dict.
    """
    dataset_id = result.get("dataset_id", "")
    rag_context = result.get("rag_analysis_context") or ""
    rag_stored = bool(result.get("rag_stored"))
    rag_summary = result.get("rag_summary") or ""
    memory_trace: list[dict[str, Any]] = result.get("memory_trace") or []

    # Derive which artifact types were stored from memory_trace
    stored_artifacts: list[str] = []
    collections: list[str] = []
    total_chunks = 0
    for evt in memory_trace:
        if evt.get("status") == "success":
            event_name = evt.get("event", "")
            if event_name == "store_profile":
                stored_artifacts.append("profile")
            elif event_name == "store_insights":
                stored_artifacts.append("insights")
            elif event_name == "store_report":
                stored_artifacts.append("report")
            col = evt.get("collection")
            if col and col not in collections:
                collections.append(col)
            total_chunks += evt.get("chunks", 0)

    # Check if memory subsystem is reachable at all
    memory_available = True
    try:
        from utils.memory import ContextStore

        ContextStore()  # smoke-test import chain
    except Exception:
        memory_available = False

    return MemoryStatus(
        dataset_id=dataset_id,
        memory_available=memory_available,
        previous_context_found=bool(rag_context),
        previous_context_chars=len(rag_context),
        current_run_stored=rag_stored,
        rag_summary=rag_summary,
        memory_trace=memory_trace,
        stored_artifacts=stored_artifacts,
        collections_used=collections,
        total_chunks_stored=total_chunks,
    )


# ---------------------------------------------------------------------------
# Retrieval helpers (live queries against ChromaDB)
# ---------------------------------------------------------------------------


def retrieve_high_priority_insights(dataset_id: str) -> RetrievalResult:
    """Fetch high-priority insights from memory for a dataset."""
    try:
        from agents.rag import RAGAgent

        rag = RAGAgent(dataset_id=dataset_id)
        text = rag.get_high_priority_insights(k=5)
        return RetrievalResult(
            label="High-priority insights",
            content=text,
            empty=not bool(text),
        )
    except Exception as exc:
        logger.debug("retrieve_high_priority_insights failed: %s", exc)
        return RetrievalResult(label="High-priority insights", content="", empty=True)


def retrieve_insights_by_category(dataset_id: str, category: str) -> RetrievalResult:
    """Fetch insights filtered by category from memory."""
    try:
        from agents.rag import RAGAgent

        rag = RAGAgent(dataset_id=dataset_id)
        text = rag.get_insights_by_category(category, k=5)
        label = f"{category.capitalize()} insights"
        return RetrievalResult(label=label, content=text, empty=not bool(text))
    except Exception as exc:
        logger.debug("retrieve_insights_by_category(%s) failed: %s", category, exc)
        return RetrievalResult(label=f"{category.capitalize()} insights", content="", empty=True)


def retrieve_analysis_context(dataset_id: str) -> RetrievalResult:
    """Fetch the full analysis context from memory (profiles + insights + reports)."""
    try:
        from agents.rag import RAGAgent

        rag = RAGAgent(dataset_id=dataset_id)
        text = rag.get_analysis_context(k=3)
        return RetrievalResult(
            label="Full analysis context",
            content=text,
            empty=not bool(text),
        )
    except Exception as exc:
        logger.debug("retrieve_analysis_context failed: %s", exc)
        return RetrievalResult(label="Full analysis context", content="", empty=True)


# ---------------------------------------------------------------------------
# Constants shared with the UI renderer
# ---------------------------------------------------------------------------

INSIGHT_CATEGORIES: list[str] = [
    "trend",
    "anomaly",
    "correlation",
    "distribution",
    "general",
]

CATEGORY_ICONS: dict[str, str] = {
    "trend": "📈",
    "anomaly": "⚠️",
    "correlation": "🔗",
    "distribution": "📊",
    "general": "💡",
}
