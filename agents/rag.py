"""
agents/rag.py — RAG (Retrieval-Augmented Generation) agent.

Retrieves relevant context from ChromaDB and formats it for injection
into other agents' prompts (Q&A, Reporter, follow-up queries).

This is NOT an LLM agent — it does not call the LLM itself.
It is a retrieval service that other agents consume.

Architecture:
    - RAGAgent (class): Retrieves and formats context from ChromaDB.
    - dataset_id: isolates context per uploaded file.
    - Structured insight search: filter by priority/category.
    - Deduplication and truncation.

Usage:
    from agents.rag import RAGAgent

    rag = RAGAgent(dataset_id="sales_2024.csv")

    # After pipeline:
    rag.save_analysis(profile_text="...", insights=[...], report_text="...")

    # For Q&A:
    context = rag.get_context_for_query("top product")

    # For follow-up:
    context = rag.get_context_for_followup("the anomalies you mentioned")

    # After Q&A exchange:
    rag.save_qa_exchange("What is the top product?", "Widget Pro.")

    # New dataset uploaded:
    rag.switch_dataset("customers_2024.csv")
"""

from __future__ import annotations

import logging
from typing import ClassVar

from utils.memory import ContextStore

logger = logging.getLogger(__name__)


class RAGAgent:
    """Retrieval agent that bridges ChromaDB and the LLM agents.

    Each instance is bound to a dataset_id to isolate context per uploaded file.
    When the user uploads a new file, call switch_dataset() or create a new instance.
    """

    DEFAULT_K: ClassVar[int] = 3
    MAX_CONTEXT_LENGTH: ClassVar[int] = 3000

    def __init__(
        self,
        dataset_id: str = "default",
        persist_dir: str | None = None,
    ):
        """
        Args:
            dataset_id: Identifier for the current dataset (e.g. filename).
            persist_dir: ChromaDB storage directory.
        """
        self.dataset_id = dataset_id
        self.store = ContextStore(persist_dir=persist_dir)

    # ── Dataset management ─────────────────────────────────────────────────

    def switch_dataset(self, dataset_id: str) -> None:
        """Switch to a different dataset context.

        Call this when the user uploads a new file.

        Args:
            dataset_id: New dataset identifier (e.g. filename).
        """
        self.dataset_id = dataset_id
        logger.info(f"RAG: switched to dataset '{dataset_id}'")

    # ── Retrieval methods ──────────────────────────────────────────────────

    def get_context_for_query(self, query: str, k: int | None = None) -> str:
        """Retrieve relevant context for a Q&A / Text-to-Code query.

        Searches all collections, filtered to the current dataset.

        Args:
            query: The user's question.
            k: Number of results per collection.

        Returns:
            Formatted context string ready for prompt injection.
        """
        k = k or self.DEFAULT_K
        chunks = self.store.search(
            query,
            collection=None,
            dataset_id=self.dataset_id,
            k=k,
        )

        if not chunks:
            return ""

        return self._format_and_truncate(chunks)

    def get_context_for_followup(self, query: str, k: int | None = None) -> str:
        """Retrieve context for follow-up queries.

        Prioritizes insights and Q&A history over profiles/reports.

        Args:
            query: The follow-up question.
            k: Number of results per collection.

        Returns:
            Formatted context string.
        """
        k = k or self.DEFAULT_K

        # Priority: insights first, then Q&A
        insight_chunks = self.store.search(
            query,
            collection=ContextStore.INSIGHTS,
            dataset_id=self.dataset_id,
            k=k,
        )
        qa_chunks = self.store.search(
            query,
            collection=ContextStore.QA_HISTORY,
            dataset_id=self.dataset_id,
            k=k,
        )

        chunks = insight_chunks + qa_chunks

        if not chunks:
            # Fallback: search everything
            chunks = self.store.search(
                query,
                collection=None,
                dataset_id=self.dataset_id,
                k=k,
            )

        if not chunks:
            return ""

        return self._format_and_truncate(chunks)

    def get_high_priority_insights(self, k: int = 5) -> str:
        """Retrieve only high-priority insights for the current dataset.

        Useful for executive summaries and report generation.

        Args:
            k: Number of results.

        Returns:
            Formatted context string of high-priority insights.
        """
        chunks = self.store.search_insights(
            query="key findings critical important",
            dataset_id=self.dataset_id,
            priority="high",
            k=k,
        )

        if not chunks:
            return ""

        return self._format_and_truncate(chunks)

    def get_insights_by_category(self, category: str, k: int = 5) -> str:
        """Retrieve insights filtered by category.

        Args:
            category: One of trend, anomaly, correlation, distribution, general.
            k: Number of results.

        Returns:
            Formatted context string.
        """
        chunks = self.store.search_insights(
            query=f"{category} findings analysis",
            dataset_id=self.dataset_id,
            category=category,
            k=k,
        )

        if not chunks:
            return ""

        return self._format_and_truncate(chunks)

    def get_analysis_context(self, k: int | None = None) -> str:
        """Retrieve full analysis context for report generation.

        Args:
            k: Number of results per collection.

        Returns:
            Formatted context from profiles + insights + reports.
        """
        k = k or self.DEFAULT_K

        profile_chunks = self.store.search(
            "dataset profile overview statistics",
            collection=ContextStore.PROFILES,
            dataset_id=self.dataset_id,
            k=k,
        )
        insight_chunks = self.store.search(
            "key insights findings analysis",
            collection=ContextStore.INSIGHTS,
            dataset_id=self.dataset_id,
            k=k,
        )
        report_chunks = self.store.search(
            "report summary recommendations",
            collection=ContextStore.REPORTS,
            dataset_id=self.dataset_id,
            k=k,
        )

        all_chunks = profile_chunks + insight_chunks + report_chunks

        if not all_chunks:
            return ""

        return self._format_and_truncate(all_chunks)

    # ── Storage methods ────────────────────────────────────────────────────

    def save_analysis(
        self,
        profile_text: str = "",
        insights: list[dict] | str = "",
        report_text: str = "",
    ) -> None:
        """Store all analysis outputs in ChromaDB.

        Call this after the pipeline completes.

        Args:
            profile_text: Profiler output markdown.
            insights: List of insight dicts (structured) or raw markdown string.
            report_text: Reporter output markdown.
        """
        if profile_text:
            count = self.store.store_profile(profile_text, dataset_id=self.dataset_id)
            logger.info(f"RAG: stored profile ({count} chunks)")

        if insights:
            count = self.store.store_insights(insights, dataset_id=self.dataset_id)
            logger.info(f"RAG: stored insights ({count} chunks)")

        if report_text:
            count = self.store.store_report(report_text, dataset_id=self.dataset_id)
            logger.info(f"RAG: stored report ({count} chunks)")

    def save_qa_exchange(self, question: str, answer: str) -> None:
        """Store a Q&A exchange.

        Args:
            question: The user's question.
            answer: The system's answer.
        """
        self.store.store_qa(question, answer, dataset_id=self.dataset_id)
        logger.info("RAG: stored Q&A exchange")

    # ── Utility ────────────────────────────────────────────────────────────

    def clear_memory(self) -> None:
        """Clear all stored context. Use when resetting the app."""
        self.store.clear()
        logger.info("RAG: all memory cleared")

    # ── Private helpers ────────────────────────────────────────────────────

    def _format_and_truncate(self, chunks: list[str]) -> str:
        """Deduplicate, join, and truncate chunks."""
        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique.append(chunk)

        text = "\n\n---\n\n".join(unique)

        # Truncate
        if len(text) <= self.MAX_CONTEXT_LENGTH:
            return text

        truncated = text[: self.MAX_CONTEXT_LENGTH]
        last_period = truncated.rfind(".")
        if last_period > self.MAX_CONTEXT_LENGTH // 2:
            truncated = truncated[: last_period + 1]

        return truncated + "\n\n[... context truncated]"
