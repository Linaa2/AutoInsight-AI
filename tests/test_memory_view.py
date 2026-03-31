"""Tests for app.memory_view — the Memory / RAG view-model layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.memory_view import (
    CATEGORY_ICONS,
    INSIGHT_CATEGORIES,
    MemoryStatus,
    RetrievalResult,
    build_memory_status,
    retrieve_analysis_context,
    retrieve_high_priority_insights,
    retrieve_insights_by_category,
)

# ---------------------------------------------------------------------------
# build_memory_status
# ---------------------------------------------------------------------------


class TestBuildMemoryStatus:
    """Tests for the pure build_memory_status function."""

    def test_empty_result(self) -> None:
        status = build_memory_status({})
        assert status.dataset_id == ""
        assert status.memory_available is True  # import chain works in test env
        assert status.previous_context_found is False
        assert status.previous_context_chars == 0
        assert status.current_run_stored is False
        assert status.rag_summary == ""
        assert status.stored_artifacts == []
        assert status.collections_used == []
        assert status.total_chunks_stored == 0

    def test_full_result(self) -> None:
        result = {
            "dataset_id": "test_sales",
            "rag_analysis_context": "Prior context from memory " * 10,
            "rag_stored": True,
            "rag_summary": "3 artifacts stored",
            "memory_trace": [
                {
                    "event": "store_profile",
                    "status": "success",
                    "collection": "profiles",
                    "chunks": 1,
                },
                {
                    "event": "store_insights",
                    "status": "success",
                    "collection": "insights",
                    "chunks": 5,
                },
                {
                    "event": "store_report",
                    "status": "success",
                    "collection": "reports",
                    "chunks": 3,
                },
            ],
        }
        status = build_memory_status(result)
        assert status.dataset_id == "test_sales"
        assert status.previous_context_found is True
        assert status.previous_context_chars > 0
        assert status.current_run_stored is True
        assert status.rag_summary == "3 artifacts stored"
        assert sorted(status.stored_artifacts) == ["insights", "profile", "report"]
        assert len(status.collections_used) == 3
        assert status.total_chunks_stored == 9

    def test_failed_events_not_in_artifacts(self) -> None:
        result = {
            "memory_trace": [
                {
                    "event": "store_profile",
                    "status": "failed",
                    "collection": "profiles",
                    "chunks": 0,
                    "message": "ChromaDB error",
                },
            ],
        }
        status = build_memory_status(result)
        assert status.stored_artifacts == []
        assert status.total_chunks_stored == 0

    def test_no_context_means_not_found(self) -> None:
        result = {"rag_analysis_context": ""}
        status = build_memory_status(result)
        assert status.previous_context_found is False
        assert status.previous_context_chars == 0

    def test_memory_unavailable_on_import_error(self) -> None:
        with (
            patch(
                "app.memory_view.ContextStore",
                side_effect=ImportError("no chromadb"),
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {"utils.memory": MagicMock(ContextStore=MagicMock(side_effect=Exception))},
            ),
        ):
            status = build_memory_status({})
            assert status.memory_available is False


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


class TestRetrievalHelpers:
    """Tests for the retrieval wrapper functions."""

    @patch("app.memory_view.RAGAgent", create=True)
    def test_retrieve_high_priority_success(self, mock_cls: MagicMock) -> None:
        mock_rag = MagicMock()
        mock_rag.get_high_priority_insights.return_value = "insight A\ninsight B"
        mock_cls.return_value = mock_rag

        # Patch the lazy import
        with patch.dict("sys.modules", {"agents.rag": MagicMock(RAGAgent=mock_cls)}):
            res = retrieve_high_priority_insights("ds1")
        assert not res.empty
        assert "insight A" in res.content

    def test_retrieve_high_priority_graceful_failure(self) -> None:
        with patch.dict(
            "sys.modules",
            {"agents.rag": MagicMock(RAGAgent=MagicMock(side_effect=Exception("boom")))},
        ):
            res = retrieve_high_priority_insights("ds1")
        assert res.empty
        assert res.content == ""

    @patch("app.memory_view.RAGAgent", create=True)
    def test_retrieve_by_category_success(self, mock_cls: MagicMock) -> None:
        mock_rag = MagicMock()
        mock_rag.get_insights_by_category.return_value = "trend data"
        mock_cls.return_value = mock_rag

        with patch.dict("sys.modules", {"agents.rag": MagicMock(RAGAgent=mock_cls)}):
            res = retrieve_insights_by_category("ds1", "trend")
        assert not res.empty
        assert "trend" in res.label.lower()

    @patch("app.memory_view.RAGAgent", create=True)
    def test_retrieve_analysis_context_success(self, mock_cls: MagicMock) -> None:
        mock_rag = MagicMock()
        mock_rag.get_analysis_context.return_value = "full context blob"
        mock_cls.return_value = mock_rag

        with patch.dict("sys.modules", {"agents.rag": MagicMock(RAGAgent=mock_cls)}):
            res = retrieve_analysis_context("ds1")
        assert not res.empty
        assert res.content == "full context blob"

    def test_retrieve_analysis_context_empty(self) -> None:
        with patch.dict(
            "sys.modules",
            {"agents.rag": MagicMock(RAGAgent=MagicMock(side_effect=Exception))},
        ):
            res = retrieve_analysis_context("ds1")
        assert res.empty


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Validate exported constants."""

    def test_categories_non_empty(self) -> None:
        assert len(INSIGHT_CATEGORIES) >= 3

    def test_all_categories_have_icons(self) -> None:
        for cat in INSIGHT_CATEGORIES:
            assert cat in CATEGORY_ICONS, f"Missing icon for {cat}"

    def test_memory_status_frozen(self) -> None:
        status = MemoryStatus()
        with pytest.raises(AttributeError):
            status.dataset_id = "x"  # type: ignore[misc]

    def test_retrieval_result_frozen(self) -> None:
        res = RetrievalResult(label="test", content="data", empty=False)
        with pytest.raises(AttributeError):
            res.empty = True  # type: ignore[misc]
