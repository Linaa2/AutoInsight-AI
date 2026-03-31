"""tests/test_langfuse_view.py — Unit tests for the LangFuse observability view model."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.langfuse_view import (
    LangfuseNodeSummary,
    LangfuseRunSummary,
    build_langfuse_run_summary,
    build_trace_url,
)

# ---------------------------------------------------------------------------
# build_trace_url
# ---------------------------------------------------------------------------


class TestBuildTraceUrl:
    def test_returns_none_if_no_host(self) -> None:
        assert build_trace_url(None, "abc123") is None

    def test_returns_none_if_no_trace_id(self) -> None:
        assert build_trace_url("http://localhost:3001", None) is None

    def test_returns_none_if_empty_trace_id(self) -> None:
        assert build_trace_url("http://localhost:3001", "") is None

    def test_builds_correct_url(self) -> None:
        url = build_trace_url("http://localhost:3001", "trace-abc")
        assert url == "http://localhost:3001/trace/trace-abc"

    def test_strips_trailing_slash_from_host(self) -> None:
        url = build_trace_url("http://localhost:3001/", "trace-abc")
        assert url == "http://localhost:3001/trace/trace-abc"

    def test_works_with_non_localhost_host(self) -> None:
        url = build_trace_url("https://langfuse.example.com", "xyz")
        assert url == "https://langfuse.example.com/trace/xyz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace_entry(
    node: str,
    status: str = "success",
    duration: float = 1.0,
    started: str = "2026-01-01T10:00:00+00:00",
    finished: str = "2026-01-01T10:00:01+00:00",
    summary: str = "",
    model: str | None = None,
    provider: str | None = None,
    error: str | None = None,
) -> dict:
    entry: dict = {
        "node": node,
        "status": status,
        "started_at": started,
        "finished_at": finished,
        "duration_s": duration,
        "summary": summary,
    }
    if model:
        entry["telemetry"] = {"model_info": {"model_name": model, "provider": provider or "ollama"}}
    if error:
        entry["error"] = error
    return entry


# ---------------------------------------------------------------------------
# build_langfuse_run_summary
# ---------------------------------------------------------------------------


class TestBuildLangfuseRunSummary:
    def test_empty_result_disabled_safe_defaults(self) -> None:
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary({})

        assert summary.enabled is False
        assert summary.trace_id is None
        assert summary.trace_url is None
        assert summary.host is None
        assert summary.node_summaries == []
        assert summary.status == "disabled"
        assert summary.nodes_succeeded == 0
        assert summary.nodes_failed == 0
        assert summary.nodes_skipped == 0
        assert summary.memory_events_count == 0
        assert summary.memory_stored is False
        assert summary.rag_context_found is False

    def test_enabled_with_trace_id_builds_url(self) -> None:
        result = {"langfuse_trace_id": "trace-xyz"}
        with (
            patch("app.langfuse_view.is_langfuse_enabled", return_value=True),
            patch("app.langfuse_view.settings") as mock_s,
        ):
            mock_s.LANGFUSE_HOST = "http://localhost:3001"
            summary = build_langfuse_run_summary(result)

        assert summary.enabled is True
        assert summary.trace_id == "trace-xyz"
        assert summary.trace_url == "http://localhost:3001/trace/trace-xyz"
        assert summary.host == "http://localhost:3001"

    def test_enabled_without_trace_id_has_no_url(self) -> None:
        with (
            patch("app.langfuse_view.is_langfuse_enabled", return_value=True),
            patch("app.langfuse_view.settings") as mock_s,
        ):
            mock_s.LANGFUSE_HOST = "http://localhost:3001"
            summary = build_langfuse_run_summary({})

        assert summary.trace_id is None
        assert summary.trace_url is None

    def test_node_summaries_from_graph_trace(self) -> None:
        result = {
            "graph_trace": [
                _trace_entry("profiler", "success", model="qwen3:14b", provider="ollama"),
                _trace_entry("analyst", "success", model="qwen3:14b"),
                _trace_entry("reporter", "failed", error="LLM timeout"),
            ]
        }
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        assert len(summary.node_summaries) == 3
        assert summary.nodes_succeeded == 2
        assert summary.nodes_failed == 1
        assert summary.nodes_skipped == 0

    def test_all_succeeded_status_is_success(self) -> None:
        result = {
            "langfuse_trace_id": "t1",
            "graph_trace": [_trace_entry("profiler"), _trace_entry("analyst")],
        }
        with (
            patch("app.langfuse_view.is_langfuse_enabled", return_value=True),
            patch("app.langfuse_view.settings") as mock_s,
        ):
            mock_s.LANGFUSE_HOST = "http://localhost:3001"
            summary = build_langfuse_run_summary(result)

        assert summary.status == "success"

    def test_mixed_success_failed_status_is_partial(self) -> None:
        result = {
            "langfuse_trace_id": "t1",
            "graph_trace": [
                _trace_entry("profiler", "success"),
                _trace_entry("analyst", "failed"),
            ],
        }
        with (
            patch("app.langfuse_view.is_langfuse_enabled", return_value=True),
            patch("app.langfuse_view.settings") as mock_s,
        ):
            mock_s.LANGFUSE_HOST = "http://localhost:3001"
            summary = build_langfuse_run_summary(result)

        assert summary.status == "partial"

    def test_all_failed_status_is_failed(self) -> None:
        result = {
            "langfuse_trace_id": "t1",
            "graph_trace": [_trace_entry("profiler", "failed")],
        }
        with (
            patch("app.langfuse_view.is_langfuse_enabled", return_value=True),
            patch("app.langfuse_view.settings") as mock_s,
        ):
            mock_s.LANGFUSE_HOST = "http://localhost:3001"
            summary = build_langfuse_run_summary(result)

        assert summary.status == "failed"

    def test_disabled_status_is_disabled_regardless_of_nodes(self) -> None:
        result = {"graph_trace": [_trace_entry("profiler", "success")]}
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        assert summary.status == "disabled"

    def test_deduplicates_nodes_keeps_last(self) -> None:
        result = {
            "graph_trace": [
                _trace_entry("profiler", "failed"),
                _trace_entry("profiler", "success"),  # retry — keep this
            ]
        }
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        assert len(summary.node_summaries) == 1
        assert summary.node_summaries[0].status == "success"

    def test_model_info_extracted_correctly(self) -> None:
        result = {"graph_trace": [_trace_entry("profiler", model="qwen3:14b", provider="ollama")]}
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        node = summary.node_summaries[0]
        assert node.model_name == "qwen3:14b"
        assert node.provider == "ollama"

    def test_no_model_info_gives_none(self) -> None:
        result = {"graph_trace": [_trace_entry("profiler")]}
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        assert summary.node_summaries[0].model_name is None
        assert summary.node_summaries[0].provider is None

    def test_memory_trace_collections_extracted(self) -> None:
        result = {
            "rag_stored": True,
            "rag_analysis_context": "some prior context",
            "rag_summary": "2 artifacts stored",
            "memory_trace": [
                {"event": "store_profile", "status": "success", "collection": "profiles"},
                {"event": "store_report", "status": "success", "collection": "reports"},
            ],
        }
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        assert summary.memory_events_count == 2
        assert summary.memory_stored is True
        assert summary.rag_context_found is True
        assert "profiles" in summary.memory_collections
        assert "reports" in summary.memory_collections
        assert summary.memory_events_summary == "2 artifacts stored"

    def test_duration_computed_from_timestamps(self) -> None:
        result = {
            "graph_trace": [
                _trace_entry(
                    "profiler",
                    started="2026-01-01T10:00:00+00:00",
                    finished="2026-01-01T10:00:02+00:00",
                    duration=2.0,
                ),
                _trace_entry(
                    "analyst",
                    started="2026-01-01T10:00:02+00:00",
                    finished="2026-01-01T10:00:05+00:00",
                    duration=3.0,
                ),
            ]
        }
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        assert summary.duration_s == 5.0
        assert summary.started_at == "2026-01-01T10:00:00+00:00"

    def test_graceful_with_missing_timestamps(self) -> None:
        """Missing started_at / finished_at must not crash."""
        result = {"graph_trace": [{"node": "profiler", "status": "success"}]}
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        assert summary.nodes_succeeded == 1
        assert summary.duration_s is None

    def test_dataset_and_file_name_forwarded(self) -> None:
        result = {"dataset_id": "sales_csv", "file_name": "sales.csv"}
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        assert summary.dataset_id == "sales_csv"
        assert summary.file_name == "sales.csv"

    def test_skipped_nodes_counted(self) -> None:
        result = {
            "graph_trace": [
                _trace_entry("analyst", "skipped"),
                _trace_entry("visualizer", "skipped"),
            ]
        }
        with patch("app.langfuse_view.is_langfuse_enabled", return_value=False):
            summary = build_langfuse_run_summary(result)

        assert summary.nodes_skipped == 2
        assert summary.nodes_succeeded == 0


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


class TestDataclassInvariants:
    def test_node_summary_is_frozen(self) -> None:
        node = LangfuseNodeSummary(
            node_name="profiler",
            status="success",
            duration_s=1.0,
            model_name=None,
            provider=None,
            summary="ok",
            error=None,
        )
        with pytest.raises(AttributeError):
            node.status = "failed"  # type: ignore[misc]

    def test_run_summary_is_frozen(self) -> None:
        summary = LangfuseRunSummary(
            enabled=False,
            host=None,
            trace_id=None,
            trace_url=None,
            dataset_id=None,
            file_name=None,
            status="disabled",
            started_at=None,
            duration_s=None,
        )
        with pytest.raises(AttributeError):
            summary.enabled = True  # type: ignore[misc]

    def test_run_summary_node_list_defaults_to_empty(self) -> None:
        summary = LangfuseRunSummary(
            enabled=False,
            host=None,
            trace_id=None,
            trace_url=None,
            dataset_id=None,
            file_name=None,
            status="disabled",
            started_at=None,
            duration_s=None,
        )
        assert summary.node_summaries == []
        assert summary.memory_collections == []
