"""
utils/memory.py — ChromaDB contextual memory for AutoInsight AI.

Improvements over basic version:
    - dataset_id metadata: isolates analyses per uploaded file
    - Structured insight storage: each insight stored individually with metadata
      (priority, category) for fine-grained retrieval

Configuration (environment variables)::

    OLLAMA_BASE_URL    = http://localhost:11434
    OLLAMA_EMBED_MODEL = nomic-embed-text
    CHROMA_DIR         = ./chroma_db

Usage:
    from utils.memory import ContextStore

    store = ContextStore()

    # Store with dataset isolation:
    store.store_profile("1500 rows...", dataset_id="sales_2024.csv")
    store.store_insights([...], dataset_id="sales_2024.csv")

    # Search within a specific dataset:
    results = store.search("anomalies", dataset_id="sales_2024.csv")

    # Search across all datasets:
    results = store.search("anomalies")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EmbeddingClient — mirrors LLMClient pattern from llm.py
# ---------------------------------------------------------------------------


class EmbeddingClient:
    """Factory for embedding models.

    Reads configuration from the centralised :data:`~config.settings.settings`
    singleton — the single source of truth for ``OLLAMA_BASE_URL`` and
    ``OLLAMA_EMBED_MODEL``.
    """

    @staticmethod
    def get_embeddings() -> Embeddings:
        """Return the configured Ollama embeddings model."""
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBED_MODEL,
        )


# ---------------------------------------------------------------------------
# ContextStore
# ---------------------------------------------------------------------------


class ContextStore:
    """ChromaDB-backed memory with dataset isolation and structured storage.

    Collections:
        - "profiles"   : dataset profile outputs
        - "insights"   : analyst-generated insights (stored individually)
        - "reports"    : reporter outputs
        - "qa_history" : Q&A exchanges

    Every stored chunk carries metadata:
        - dataset_id : identifies which uploaded file the data belongs to
        - source     : which agent produced it (profiler, analyst, reporter, qa)
        - priority   : (insights only) high / medium / low
        - category   : (insights only) trend / anomaly / correlation / etc.
    """

    PROFILES: ClassVar[str] = "profiles"
    INSIGHTS: ClassVar[str] = "insights"
    REPORTS: ClassVar[str] = "reports"
    QA_HISTORY: ClassVar[str] = "qa_history"

    ALL_COLLECTIONS: ClassVar[list[str]] = [
        "profiles",
        "insights",
        "reports",
        "qa_history",
    ]

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.CHROMA_DIR
        self._embeddings = EmbeddingClient.get_embeddings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    # ── Store methods ──────────────────────────────────────────────────────

    def store_profile(self, text: str, dataset_id: str = "default") -> int:
        """Store a dataset profile.

        Args:
            text: Profiler output markdown.
            dataset_id: Identifier for the uploaded file (e.g. filename).

        Returns:
            Number of chunks stored.
        """
        metadata = {"source": "profiler", "dataset_id": dataset_id}
        return self._store(text, collection=self.PROFILES, metadata=metadata)

    def store_insights(
        self,
        insights: list[dict] | str,
        dataset_id: str = "default",
    ) -> int:
        """Store insights — either as structured dicts or raw markdown.

        If a list of dicts is provided, each insight is stored individually
        with its priority and category as metadata for fine-grained retrieval.

        Args:
            insights: List of insight dicts or raw markdown string.
            dataset_id: Identifier for the uploaded file.

        Returns:
            Number of chunks stored.
        """
        if isinstance(insights, str):
            metadata = {"source": "analyst", "dataset_id": dataset_id}
            return self._store(insights, collection=self.INSIGHTS, metadata=metadata)

        # Structured storage: one entry per insight
        total = 0
        for insight in insights:
            text = (
                f"Title: {insight.get('title', 'Untitled')}\n"
                f"Observation: {insight.get('observation', '')}\n"
                f"Hypothesis: {insight.get('hypothesis', '')}\n"
                f"Recommendation: {insight.get('recommendation', '')}"
            )
            metadata = {
                "source": "analyst",
                "dataset_id": dataset_id,
                "priority": insight.get("priority", "medium"),
                "category": insight.get("category", "general"),
                "title": insight.get("title", "Untitled"),
            }
            total += self._store(text, collection=self.INSIGHTS, metadata=metadata)

        logger.info(f"Stored {len(insights)} insights ({total} chunks) for '{dataset_id}'")
        return total

    def store_report(self, text: str, dataset_id: str = "default") -> int:
        """Store a reporter output.

        Args:
            text: Reporter output markdown.
            dataset_id: Identifier for the uploaded file.

        Returns:
            Number of chunks stored.
        """
        metadata = {"source": "reporter", "dataset_id": dataset_id}
        return self._store(text, collection=self.REPORTS, metadata=metadata)

    def store_qa(
        self,
        question: str,
        answer: str,
        dataset_id: str = "default",
    ) -> int:
        """Store a Q&A exchange.

        Args:
            question: The user's question.
            answer: The system's answer.
            dataset_id: Identifier for the uploaded file.

        Returns:
            Number of chunks stored.
        """
        text = f"Question: {question}\nAnswer: {answer}"
        metadata = {"source": "qa", "dataset_id": dataset_id}
        return self._store(text, collection=self.QA_HISTORY, metadata=metadata)

    # ── Search methods ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        collection: str | None = None,
        dataset_id: str | None = None,
        k: int = 3,
    ) -> list[str]:
        """Search for relevant chunks.

        Args:
            query: Search query.
            collection: Collection to search. If None, searches all.
            dataset_id: Filter by dataset. If None, searches across all datasets.
            k: Number of results per collection.

        Returns:
            List of relevant text chunks.
        """
        where_filter = {"dataset_id": dataset_id} if dataset_id else None

        if collection:
            return self._search_collection(query, collection, k, where_filter)

        results: list[str] = []
        for col in self.ALL_COLLECTIONS:
            results.extend(self._search_collection(query, col, k, where_filter))
        return results

    def search_insights(
        self,
        query: str,
        dataset_id: str | None = None,
        priority: str | None = None,
        category: str | None = None,
        k: int = 5,
    ) -> list[str]:
        """Search insights with optional metadata filters.

        Args:
            query: Search query.
            dataset_id: Filter by dataset.
            priority: Filter by priority (high/medium/low).
            category: Filter by category (trend/anomaly/correlation/distribution/general).
            k: Number of results.

        Returns:
            List of relevant insight text chunks.
        """
        conditions: dict[str, str] = {}
        if dataset_id:
            conditions["dataset_id"] = dataset_id
        if priority:
            conditions["priority"] = priority
        if category:
            conditions["category"] = category

        # ChromaDB rejects a flat dict with >1 key ("Expected where to have
        # exactly one operator").  Use $and when multiple conditions are needed.
        if not conditions:
            chroma_filter: dict | None = None
        elif len(conditions) == 1:
            chroma_filter = conditions
        else:
            chroma_filter = {"$and": [{key: val} for key, val in conditions.items()]}

        return self._search_collection(
            query,
            self.INSIGHTS,
            k,
            chroma_filter,
        )

    def search_as_text(
        self,
        query: str,
        collection: str | None = None,
        dataset_id: str | None = None,
        k: int = 3,
    ) -> str:
        """Search and return results as a single joined string.

        Args:
            query: Search query.
            collection: Collection to search. If None, searches all.
            dataset_id: Filter by dataset.
            k: Number of results per collection.

        Returns:
            Concatenated text of relevant chunks, or empty string.
        """
        chunks = self.search(query, collection, dataset_id, k)
        if not chunks:
            return ""
        return "\n\n---\n\n".join(chunks)

    # ── Utility methods ────────────────────────────────────────────────────

    def clear(self, collection: str | None = None) -> None:
        """Delete all data from one or all collections."""
        import chromadb

        client = chromadb.PersistentClient(path=self.persist_dir)
        to_clear = [collection] if collection else self.ALL_COLLECTIONS

        for col_name in to_clear:
            try:
                client.delete_collection(col_name)
                logger.info(f"Cleared collection '{col_name}'")
            except Exception:
                pass

    def clear_dataset(self, dataset_id: str) -> None:
        """Delete all stored chunks for *dataset_id* across every collection.

        Called by :func:`~orchestration.graph.rag_storage_node` **before** writing
        fresh data for a run so that re-analysing the same file replaces the
        previous run's chunks rather than accumulating duplicates indefinitely.

        Clearing happens *after* :func:`~orchestration.graph.reporter_node` has
        already retrieved the previous run's context, so the enrichment path is
        unaffected.
        """
        import chromadb

        client = chromadb.PersistentClient(path=self.persist_dir)
        for col_name in self.ALL_COLLECTIONS:
            try:
                col = client.get_collection(col_name)
                # get() returns {"ids": [...], ...} — fetch IDs matching this dataset
                existing = col.get(where={"dataset_id": dataset_id})
                ids_to_delete = existing.get("ids") or []
                if ids_to_delete:
                    col.delete(ids=ids_to_delete)
                    logger.info(
                        "clear_dataset: removed %d chunks from '%s' for dataset '%s'",
                        len(ids_to_delete),
                        col_name,
                        dataset_id,
                    )
            except Exception as exc:
                # Collection may not exist yet on the first ever run — safe to skip.
                logger.debug("clear_dataset: could not clear '%s': %s", col_name, exc)

    def list_collections(self) -> list[str]:
        """List all existing collections."""
        import chromadb

        client = chromadb.PersistentClient(path=self.persist_dir)
        return [c.name for c in client.list_collections()]

    @staticmethod
    def make_dataset_id(filename: str) -> str:
        """Generate a stable dataset_id from a filename.

        Uses the filename directly — simple and readable.
        """
        return filename.strip()

    # ── Private helpers ────────────────────────────────────────────────────

    def _store(self, text: str, collection: str, metadata: dict | None = None) -> int:
        """Split text into chunks and store with metadata."""
        if not text or not text.strip():
            logger.warning(f"Empty text, nothing stored in '{collection}'")
            return 0

        chunks = self._splitter.split_text(text)

        # Each chunk gets the same metadata
        metadatas = [metadata.copy() for _ in chunks] if metadata else None

        Chroma.from_texts(
            texts=chunks,
            embedding=self._embeddings,
            collection_name=collection,
            persist_directory=self.persist_dir,
            metadatas=metadatas,
        )

        logger.info(f"Stored {len(chunks)} chunks in '{collection}'")
        return len(chunks)

    def _search_collection(
        self,
        query: str,
        collection: str,
        k: int,
        where_filter: dict | None = None,
    ) -> list[str]:
        """Search a single collection with optional metadata filter."""
        try:
            vectorstore = Chroma(
                collection_name=collection,
                embedding_function=self._embeddings,
                persist_directory=self.persist_dir,
            )

            kwargs: dict = {"query": query, "k": k}
            if where_filter:
                kwargs["filter"] = where_filter

            docs = vectorstore.similarity_search(**kwargs)
            return [doc.page_content for doc in docs]

        except Exception as e:
            logger.debug(f"Search in '{collection}' failed: {e}")
            return []
