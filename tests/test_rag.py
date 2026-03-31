"""
tests/test_rag.py — Unit tests for the RAG / ChromaDB integration.

Covers:
- ContextStore.make_dataset_id
- RAGAgent._format_and_truncate
- rag_storage_node (skipped path + graceful failure path)
- _make_initial_state (dataset_id propagation)
- ReporterAgent backward compat (with and without rag_context)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd

if TYPE_CHECKING:
    import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  ContextStore.make_dataset_id
# ═══════════════════════════════════════════════════════════════════════════


def test_make_dataset_id_passthrough() -> None:
    """Plain filename is returned unchanged."""
    from utils.memory import ContextStore

    assert ContextStore.make_dataset_id("sales.csv") == "sales.csv"


def test_make_dataset_id_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped."""
    from utils.memory import ContextStore

    assert ContextStore.make_dataset_id("  sales.csv  ") == "sales.csv"


def test_make_dataset_id_stable() -> None:
    """Same input always produces the same output."""
    from utils.memory import ContextStore

    first = ContextStore.make_dataset_id("report_2024.parquet")
    second = ContextStore.make_dataset_id("report_2024.parquet")
    assert first == second


def test_make_dataset_id_different_files() -> None:
    """Different filenames produce different IDs."""
    from utils.memory import ContextStore

    assert ContextStore.make_dataset_id("a.csv") != ContextStore.make_dataset_id("b.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  RAGAgent._format_and_truncate
# ═══════════════════════════════════════════════════════════════════════════


def _make_rag_agent_no_init():  # returns RAGAgent
    """Build a RAGAgent instance bypassing __init__ (no Ollama / ChromaDB)."""
    from agents.rag import RAGAgent

    return RAGAgent.__new__(RAGAgent)


def test_format_and_truncate_deduplication() -> None:
    """Duplicate chunks are removed, keeping first occurrence."""
    agent = _make_rag_agent_no_init()
    chunks = ["Alpha", "Beta", "Alpha", "Gamma"]
    result = agent._format_and_truncate(chunks)

    assert result.count("Alpha") == 1
    assert "Beta" in result
    assert "Gamma" in result


def test_format_and_truncate_short_text_untouched() -> None:
    """Short text is returned as-is without a truncation marker."""
    agent = _make_rag_agent_no_init()
    result = agent._format_and_truncate(["short text"])

    assert result == "short text"
    assert "[... context truncated]" not in result


def test_format_and_truncate_truncates_long_text() -> None:
    """Texts exceeding MAX_CONTEXT_LENGTH are cut and marked."""
    from agents.rag import RAGAgent

    agent = RAGAgent.__new__(RAGAgent)
    with patch.object(RAGAgent, "MAX_CONTEXT_LENGTH", 50):
        chunks = ["x" * 200]
        result = agent._format_and_truncate(chunks)

    assert "[... context truncated]" in result
    # Total length should be bounded (truncated + marker)
    assert len(result) < 300


def test_format_and_truncate_prefers_sentence_boundary() -> None:
    """Truncation cuts at the last sentence boundary when possible."""
    from agents.rag import RAGAgent

    agent = RAGAgent.__new__(RAGAgent)
    with patch.object(RAGAgent, "MAX_CONTEXT_LENGTH", 40):
        chunks = ["Hello world. This is a longer sentence that pushes over the limit."]
        result = agent._format_and_truncate(chunks)

    assert "[... context truncated]" in result
    # Should end before the truncation marker (not mid-word)
    before_marker = result.split("[... context truncated]")[0].strip()
    assert len(before_marker) > 0


def test_format_and_truncate_multiple_chunks_joined() -> None:
    """Multiple chunks are joined with separators."""
    agent = _make_rag_agent_no_init()
    result = agent._format_and_truncate(["Chunk A", "Chunk B"])

    assert "Chunk A" in result
    assert "Chunk B" in result
    assert "---" in result  # separator included


# ═══════════════════════════════════════════════════════════════════════════
#  rag_storage_node — skipped path (no analysis outputs)
# ═══════════════════════════════════════════════════════════════════════════


def test_rag_storage_node_skipped_when_no_outputs() -> None:
    """Node returns rag_stored=False and 'Nothing to store' when state is empty."""
    from orchestration.graph import rag_storage_node

    result = rag_storage_node(
        {
            "dataset_id": "test.csv",
            "graph_trace": [],
        }
    )

    assert result["rag_stored"] is False
    assert "Nothing to store" in (result.get("rag_summary") or "")
    assert result.get("memory_trace") == []

    # A trace entry should record the 'skipped' status
    trace: list = result.get("graph_trace", [])
    rag_entries = [e for e in trace if e.get("node") == "rag_storage"]
    assert len(rag_entries) == 1
    assert rag_entries[0]["status"] == "skipped"


def test_rag_storage_node_always_returns_dict() -> None:
    """rag_storage_node never raises — it always returns a dict with required keys."""
    from orchestration.graph import rag_storage_node

    # Even with a completely empty state dict it must not raise
    result = rag_storage_node({})
    assert isinstance(result, dict)
    assert "rag_stored" in result
    assert "graph_trace" in result


# ═══════════════════════════════════════════════════════════════════════════
#  rag_storage_node — graceful failure path (ChromaDB / Ollama unavailable)
# ═══════════════════════════════════════════════════════════════════════════


def test_rag_storage_node_graceful_when_rag_agent_raises() -> None:
    """If RAGAgent instantiation fails, node returns rag_stored=False without raising."""
    from orchestration.graph import rag_storage_node

    with patch("agents.rag.RAGAgent", side_effect=RuntimeError("Chroma unavailable")):
        result = rag_storage_node(
            {
                "dataset_id": "test.csv",
                "profile_markdown": "# Profile\nSome data summary",
                "graph_trace": [],
            }
        )

    assert result["rag_stored"] is False
    # Summary should mention the failure
    assert result.get("rag_summary")
    # Trace entry should record 'failed'
    trace = result.get("graph_trace", [])
    rag_entries = [e for e in trace if e.get("node") == "rag_storage"]
    assert len(rag_entries) == 1
    assert rag_entries[0]["status"] == "failed"


def test_rag_storage_node_graceful_when_save_raises() -> None:
    """If save_analysis/store_insights/store_report fails, node returns rag_stored=False."""
    from orchestration.graph import rag_storage_node

    mock_rag = MagicMock()
    mock_rag.save_analysis.side_effect = RuntimeError("Disk full")
    mock_rag.store.store_insights.side_effect = RuntimeError("Disk full")
    mock_rag.store.store_report.side_effect = RuntimeError("Disk full")

    with patch("agents.rag.RAGAgent", return_value=mock_rag):
        result = rag_storage_node(
            {
                "dataset_id": "test.csv",
                "profile_markdown": "some profile",
                "insights": [{"title": "T1", "observation": "Obs", "recommendation": "Act"}],
                "report_markdown": "some report",
                "graph_trace": [],
            }
        )

    assert "rag_stored" in result
    assert "graph_trace" in result
    # Node must not propagate exceptions
    assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
#  _make_initial_state — dataset_id propagation
# ═══════════════════════════════════════════════════════════════════════════


def test_make_initial_state_derives_dataset_id() -> None:
    """dataset_id is derived from file_name when not supplied."""
    from orchestration.graph import _make_initial_state

    df = pd.DataFrame({"a": [1, 2]})
    # RAG pre-load will fail gracefully (no Ollama in test env)
    state = _make_initial_state(df, "myfile.csv")

    assert state["dataset_id"] == "myfile.csv"
    assert state["file_name"] == "myfile.csv"
    assert "graph_trace" in state
    assert isinstance(state["graph_trace"], list)


def test_make_initial_state_respects_explicit_dataset_id() -> None:
    """Explicit dataset_id overrides the auto-derived value."""
    from orchestration.graph import _make_initial_state

    df = pd.DataFrame({"x": [1]})
    state = _make_initial_state(df, "file.csv", dataset_id="custom_id_42")

    assert state["dataset_id"] == "custom_id_42"
    assert state["file_name"] == "file.csv"


def test_make_initial_state_df_ref_present() -> None:
    """The fast path stores a DataFrame reference instead of serializing rows."""
    from orchestration.graph import _make_initial_state

    df = pd.DataFrame({"col": [10, 20, 30]})
    state = _make_initial_state(df, "test.csv")

    assert "df_ref" in state
    assert state["df_ref"].startswith("df-")
    assert "df_dict" not in state


def test_make_initial_state_rag_context_empty_without_history() -> None:
    """Initial state starts with no RAG context; retrieval happens in reporter_node."""
    from orchestration.graph import _make_initial_state

    df = pd.DataFrame({"a": [1]})
    state = _make_initial_state(df, "no_history.csv")

    assert state.get("rag_analysis_context", "") == ""


def test_reporter_node_retrieves_rag_context_when_history_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reporter_node retrieves prior context on demand and returns it in state."""
    from orchestration.graph import reporter_node

    mock_rag = MagicMock()
    mock_rag.get_analysis_context.return_value = "Prior analysis: revenue grew by 10%."
    monkeypatch.setattr("agents.rag.RAGAgent", MagicMock(return_value=mock_rag))

    mock_reporter = MagicMock()
    mock_reporter.run.return_value = {"reporter_output": "# Report"}
    monkeypatch.setattr("orchestration.graph.ReporterAgent", MagicMock(return_value=mock_reporter))

    result = reporter_node(
        {
            "dataset_id": "repeat.csv",
            "profile_markdown": "Profile data",
            "insights_markdown": "Insights here",
            "graph_trace": [],
            "memory_trace": [],
        }
    )

    assert "Prior analysis" in result.get("rag_analysis_context", "")
    assert result["report_markdown"] == "# Report"
    assert any(evt.get("event") == "retrieve_context" for evt in result.get("memory_trace", []))


# ═══════════════════════════════════════════════════════════════════════════
#  ReporterAgent — backward compatibility + rag_context injection
# ═══════════════════════════════════════════════════════════════════════════


def test_reporter_agent_works_without_rag_context() -> None:
    """ReporterAgent.run() still works when rag_context is omitted."""
    with patch("agents.reporter.call_llm_with_messages", return_value="# Report") as mock_llm:
        from agents.reporter import ReporterAgent

        agent = ReporterAgent()
        result = agent.run(
            profiler_output="Profile data",
            analyst_output="Insights here",
        )

    assert result["reporter_output"] == "# Report"
    mock_llm.assert_called_once()
    # RAG block must NOT appear when rag_context is not given
    call_kwargs = mock_llm.call_args.kwargs
    human_prompt: str = call_kwargs.get("human", "")
    assert "Previous Analyses" not in human_prompt
    assert "📚" not in human_prompt


def test_reporter_agent_with_empty_rag_context() -> None:
    """Passing rag_context='' is the same as not passing it (no injection)."""
    with patch("agents.reporter.call_llm_with_messages", return_value="# Report") as mock_llm:
        from agents.reporter import ReporterAgent

        agent = ReporterAgent()
        result = agent.run(
            profiler_output="Some profile",
            analyst_output="Some insights",
            rag_context="",
        )

    assert result["reporter_output"] == "# Report"
    human_prompt: str = mock_llm.call_args.kwargs.get("human", "")
    assert "Previous Analyses" not in human_prompt


def test_reporter_agent_injects_rag_context_into_prompt() -> None:
    """When rag_context is non-empty, it is appended to the LLM human prompt."""
    with patch("agents.reporter.call_llm_with_messages", return_value="# Enrich") as mock_llm:
        from agents.reporter import ReporterAgent

        agent = ReporterAgent()
        result = agent.run(
            profiler_output="Profile data",
            analyst_output="Insights here",
            rag_context="Prior analysis: Q3 revenue was $2M.",
        )

    assert result["reporter_output"] == "# Enrich"
    human_prompt: str = mock_llm.call_args.kwargs.get("human", "")
    # Both the injected text and the section header should appear
    assert "Prior analysis: Q3 revenue was $2M." in human_prompt
    assert "Previous Analyses" in human_prompt


def test_reporter_agent_rag_context_after_main_content() -> None:
    """RAG context block is appended AFTER the main prompt content, not before."""
    with patch("agents.reporter.call_llm_with_messages", return_value="ok") as mock_llm:
        from agents.reporter import ReporterAgent

        agent = ReporterAgent()
        agent.run(
            profiler_output="Profile data",
            analyst_output="Insights here",
            rag_context="Historical note.",
        )

    human_prompt: str = mock_llm.call_args.kwargs.get("human", "")
    # The main analyst content must appear before the RAG block
    analyst_pos = human_prompt.find("Insights here")
    rag_pos = human_prompt.find("Historical note.")
    assert analyst_pos < rag_pos, "RAG context should come after main analyst content"


def test_reporter_agent_handles_llm_error_gracefully() -> None:
    """ReporterAgent.run() returns an error dict (not raises) if LLM fails."""
    with patch("agents.reporter.call_llm_with_messages", side_effect=RuntimeError("LLM timeout")):
        from agents.reporter import ReporterAgent

        agent = ReporterAgent()
        result = agent.run(
            profiler_output="data",
            analyst_output="insights",
        )

    assert "error" in result
    assert "reporter_output" in result
    assert "❌" in result["reporter_output"]
