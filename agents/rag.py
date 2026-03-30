"""
agents/rag.py — RAG (Retrieval-Augmented Generation) agent.

Retrieves relevant context from ChromaDB and formats it for injection
into other agents' prompts (Q&A, Reporter, follow-up queries).

This is NOT an LLM agent — it does not call the LLM itself.
It is a retrieval service that other agents consume.

Architecture:
    - RAGAgent (class): Retrieves and formats context from ChromaDB.
    - Prompt builders: Format retrieved context for specific use cases.

Usage:
    from agents.rag import RAGAgent

    rag = RAGAgent()

    # For Q&A / Text-to-Code: enrich the prompt with relevant context
    context = rag.get_context_for_query("What is the top product?")

    # For Reporter: get all analysis context
    context = rag.get_analysis_context()

    # For follow-up: find specific past insights
    context = rag.get_context_for_followup("the anomalies you mentioned")

    # After any Q&A exchange: save it for future retrieval
    rag.save_qa_exchange("What is the top product?", "Widget Pro.")
"""

import logging
from typing import ClassVar

from utils.memory import ContextStore

logger = logging.getLogger(__name__)


class RAGAgent:
    """Retrieval agent that bridges ChromaDB and the LLM agents.

    Responsibilities:
        1. Retrieve relevant past context for a user query.
        2. Format the context into a string ready for prompt injection.
        3. Store new analysis outputs and Q&A exchanges.
        4. Manage what context goes where (profiles, insights, reports, Q&A).
    """

    # How many chunks to retrieve per collection
    DEFAULT_K: ClassVar[int] = 3

    # Max characters of context to inject (avoid overloading the prompt)
    MAX_CONTEXT_LENGTH: ClassVar[int] = 3000

    def __init__(self, persist_dir: str | None = None):
        """
        Args:
            persist_dir: ChromaDB storage directory (passed to ContextStore).
        """
        self.store = ContextStore(persist_dir=persist_dir)

    # ── Retrieval methods (used by other agents) ───────────────────────────

    def get_context_for_query(self, query: str, k: int | None = None) -> str:
        """Retrieve relevant context for a Q&A / Text-to-Code query.

        Searches all collections (profiles, insights, reports, Q&A history)
        and returns a formatted string ready for prompt injection.

        Args:
            query: The user's question.
            k: Number of results per collection.

        Returns:
            Formatted context string, or empty string if nothing found.
        """
        k = k or self.DEFAULT_K
        chunks = self.store.search(query, collection=None, k=k)

        if not chunks:
            return ""

        context = self._format_chunks(chunks)
        return self._truncate(context)

    def get_context_for_followup(self, query: str, k: int | None = None) -> str:
        """Retrieve context specifically for follow-up queries.

        Prioritizes insights and Q&A history over profiles.

        Args:
            query: The follow-up question (e.g. "the anomalies you mentioned").
            k: Number of results per collection.

        Returns:
            Formatted context string.
        """
        k = k or self.DEFAULT_K

        # Prioritize insights and Q&A
        insight_chunks = self.store.search(query, collection=ContextStore.INSIGHTS, k=k)
        qa_chunks = self.store.search(query, collection=ContextStore.QA_HISTORY, k=k)

        chunks = insight_chunks + qa_chunks

        if not chunks:
            # Fallback to all collections
            chunks = self.store.search(query, collection=None, k=k)

        if not chunks:
            return ""

        context = self._format_chunks(chunks)
        return self._truncate(context)

    def get_analysis_context(self, k: int | None = None) -> str:
        """Retrieve the full analysis context for report generation.

        Searches profiles, insights, and reports collections.

        Args:
            k: Number of results per collection.

        Returns:
            Formatted context string.
        """
        k = k or self.DEFAULT_K

        profile_chunks = self.store.search(
            "dataset profile overview statistics",
            collection=ContextStore.PROFILES,
            k=k,
        )
        insight_chunks = self.store.search(
            "key insights findings analysis",
            collection=ContextStore.INSIGHTS,
            k=k,
        )
        report_chunks = self.store.search(
            "report summary recommendations",
            collection=ContextStore.REPORTS,
            k=k,
        )

        all_chunks = profile_chunks + insight_chunks + report_chunks

        if not all_chunks:
            return ""

        context = self._format_chunks(all_chunks)
        return self._truncate(context)

    # ── Storage methods (called after pipeline runs) ───────────────────────

    def save_analysis(
        self,
        profile_text: str = "",
        insights_text: str = "",
        report_text: str = "",
    ) -> None:
        """Store all analysis outputs in ChromaDB.

        Call this after the pipeline (profiler → analyst → reporter) completes.

        Args:
            profile_text: Profiler output markdown.
            insights_text: Analyst output markdown.
            report_text: Reporter output markdown.
        """
        if profile_text:
            count = self.store.store_profile(profile_text)
            logger.info(f"RAG: stored profile ({count} chunks)")

        if insights_text:
            count = self.store.store_insights(insights_text)
            logger.info(f"RAG: stored insights ({count} chunks)")

        if report_text:
            count = self.store.store_report(report_text)
            logger.info(f"RAG: stored report ({count} chunks)")

    def save_qa_exchange(self, question: str, answer: str) -> None:
        """Store a Q&A exchange for future retrieval.

        Call this after each Q&A interaction in the chat.

        Args:
            question: The user's question.
            answer: The system's answer.
        """
        self.store.store_qa(question, answer)
        logger.info("RAG: stored Q&A exchange")

    # ── Utility ────────────────────────────────────────────────────────────

    def clear_memory(self) -> None:
        """Clear all stored context. Use when a new dataset is uploaded."""
        self.store.clear()
        logger.info("RAG: all memory cleared")

    # ── Private helpers ────────────────────────────────────────────────────

    def _format_chunks(self, chunks: list[str]) -> str:
        """Format retrieved chunks into a readable context block."""
        if not chunks:
            return ""

        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique.append(chunk)

        return "\n\n---\n\n".join(unique)

    def _truncate(self, text: str) -> str:
        """Truncate context to avoid overloading the LLM prompt."""
        if len(text) <= self.MAX_CONTEXT_LENGTH:
            return text

        truncated = text[: self.MAX_CONTEXT_LENGTH]
        # Cut at last complete sentence
        last_period = truncated.rfind(".")
        if last_period > self.MAX_CONTEXT_LENGTH // 2:
            truncated = truncated[: last_period + 1]

        return truncated + "\n\n[... context truncated]"
