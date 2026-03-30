"""
tests/test_rag.py — RAGAgent unit tests.

Tests the RAGAgent class from agents/rag.py.
Requires Ollama running with nomic-embed-text for integration tests.

Run from the project root:
    uv run pytest tests/test_rag.py -v
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from agents.rag import RAGAgent

# ═══════════════════════════════════════════════════════════════════════════
#  Unit tests (no Ollama needed — ContextStore is mocked)
# ═══════════════════════════════════════════════════════════════════════════


class TestRAGAgentUnit:
    """Unit tests with mocked ContextStore."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch("agents.rag.ContextStore") as mock_cs:
            self.mock_store = MagicMock()
            mock_cs.return_value = self.mock_store
            self.rag = RAGAgent(dataset_id="test.csv")
            yield

    def test_init_default(self):
        """RAGAgent initializes with dataset_id."""
        assert self.rag.dataset_id == "test.csv"

    def test_switch_dataset(self):
        """switch_dataset updates dataset_id."""
        self.rag.switch_dataset("new_file.csv")
        assert self.rag.dataset_id == "new_file.csv"

    def test_get_context_for_query_with_results(self):
        """get_context_for_query returns formatted context."""
        self.mock_store.search.return_value = ["chunk1", "chunk2"]
        result = self.rag.get_context_for_query("top product")
        assert "chunk1" in result
        assert "chunk2" in result
        self.mock_store.search.assert_called_once_with(
            "top product",
            collection=None,
            dataset_id="test.csv",
            k=3,
        )

    def test_get_context_for_query_empty(self):
        """get_context_for_query returns empty string when nothing found."""
        self.mock_store.search.return_value = []
        result = self.rag.get_context_for_query("unknown")
        assert result == ""

    def test_get_context_for_followup_prioritizes_insights(self):
        """get_context_for_followup searches insights and QA first."""
        self.mock_store.search.side_effect = [
            ["insight chunk"],  # insights collection
            ["qa chunk"],  # qa_history collection
        ]
        result = self.rag.get_context_for_followup("anomalies")
        assert "insight chunk" in result
        assert "qa chunk" in result

    def test_get_context_for_followup_fallback(self):
        """get_context_for_followup falls back to all collections."""
        self.mock_store.search.side_effect = [
            [],  # insights — empty
            [],  # qa_history — empty
            ["fallback chunk"],  # all collections
        ]
        result = self.rag.get_context_for_followup("something")
        assert "fallback chunk" in result

    def test_get_high_priority_insights(self):
        """get_high_priority_insights filters by priority=high."""
        self.mock_store.search_insights.return_value = ["critical finding"]
        result = self.rag.get_high_priority_insights()
        assert "critical finding" in result
        self.mock_store.search_insights.assert_called_once_with(
            query="key findings critical important",
            dataset_id="test.csv",
            priority="high",
            k=5,
        )

    def test_get_insights_by_category(self):
        """get_insights_by_category filters by category."""
        self.mock_store.search_insights.return_value = ["trend data"]
        result = self.rag.get_insights_by_category("trend")
        assert "trend data" in result

    def test_get_analysis_context(self):
        """get_analysis_context combines profiles + insights + reports."""
        self.mock_store.search.side_effect = [
            ["profile data"],
            ["insight data"],
            ["report data"],
        ]
        result = self.rag.get_analysis_context()
        assert "profile data" in result
        assert "insight data" in result
        assert "report data" in result

    def test_save_analysis_profile(self):
        """save_analysis stores profile text."""
        self.mock_store.store_profile.return_value = 2
        self.rag.save_analysis(profile_text="Profile markdown")
        self.mock_store.store_profile.assert_called_once_with(
            "Profile markdown",
            dataset_id="test.csv",
        )

    def test_save_analysis_insights_as_list(self):
        """save_analysis stores structured insight dicts."""
        insights = [{"title": "A", "observation": "B"}]
        self.mock_store.store_insights.return_value = 1
        self.rag.save_analysis(insights=insights)
        self.mock_store.store_insights.assert_called_once_with(
            insights,
            dataset_id="test.csv",
        )

    def test_save_analysis_insights_as_string(self):
        """save_analysis stores insights as raw markdown string."""
        self.mock_store.store_insights.return_value = 1
        self.rag.save_analysis(insights="Raw insights markdown")
        self.mock_store.store_insights.assert_called_once_with(
            "Raw insights markdown",
            dataset_id="test.csv",
        )

    def test_save_analysis_report(self):
        """save_analysis stores report text."""
        self.mock_store.store_report.return_value = 3
        self.rag.save_analysis(report_text="Full report")
        self.mock_store.store_report.assert_called_once_with(
            "Full report",
            dataset_id="test.csv",
        )

    def test_save_qa_exchange(self):
        """save_qa_exchange stores question and answer."""
        self.mock_store.store_qa.return_value = 1
        self.rag.save_qa_exchange("What?", "Answer.")
        self.mock_store.store_qa.assert_called_once_with(
            "What?",
            "Answer.",
            dataset_id="test.csv",
        )

    def test_clear_memory(self):
        """clear_memory calls store.clear()."""
        self.rag.clear_memory()
        self.mock_store.clear.assert_called_once()

    def test_deduplication(self):
        """Duplicate chunks are removed."""
        self.mock_store.search.return_value = ["same", "same", "different"]
        result = self.rag.get_context_for_query("test")
        assert result.count("same") == 1
        assert "different" in result

    def test_truncation(self):
        """Long context is truncated."""
        long_text = "A" * 5000
        self.mock_store.search.return_value = [long_text]
        result = self.rag.get_context_for_query("test")
        assert len(result) <= RAGAgent.MAX_CONTEXT_LENGTH + 50  # +50 for suffix


# ═══════════════════════════════════════════════════════════════════════════
#  Integration tests (require Ollama + nomic-embed-text)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_rag_store_and_retrieve():
    """Full RAGAgent pipeline: store → search → retrieve."""
    tmp = tempfile.mkdtemp()
    try:
        rag = RAGAgent(dataset_id="integration_test.csv", persist_dir=tmp)

        rag.save_analysis(
            profile_text="Dataset has 1500 rows and 8 columns about e-commerce sales.",
            insights="Widget Pro dominates sales with 185 orders.",
            report_text="The analysis reveals strong regional concentration.",
        )

        context = rag.get_context_for_query("top selling product")
        assert len(context) > 0

        rag.save_qa_exchange("What is the best product?", "Widget Pro.")
        followup = rag.get_context_for_followup("the product you mentioned")
        assert len(followup) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.integration
def test_rag_dataset_isolation():
    """Verify dataset_id isolates context between datasets."""
    tmp = tempfile.mkdtemp()
    try:
        rag1 = RAGAgent(dataset_id="sales.csv", persist_dir=tmp)
        rag1.save_analysis(profile_text="Sales data with 1500 rows.")

        rag2 = RAGAgent(dataset_id="customers.csv", persist_dir=tmp)
        rag2.save_analysis(profile_text="Customer data with 500 rows.")

        ctx1 = rag1.get_context_for_query("rows")
        ctx2 = rag2.get_context_for_query("rows")

        assert len(ctx1) > 0
        assert len(ctx2) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
