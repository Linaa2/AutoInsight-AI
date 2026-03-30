"""
utils/memory.py — ChromaDB contextual memory for AutoInsight AI.

Stores and retrieves analysis context to enrich conversations.

Three types of data are indexed:
    1. Dataset profiles (metadata, statistics, column descriptions)
    2. Generated insights (analyst observations)
    3. Q&A history (previous questions and answers)

This module handles STORAGE ONLY. The RAG agent that uses this
context lives in agents/rag.py (Phase 5 step 2).

Usage:
    from utils.memory import ContextStore

    store = ContextStore()

    # After pipeline runs:
    store.store_profile("The dataset has 1500 rows...")
    store.store_insights("Widget Pro dominates sales...")
    store.store_report("Executive summary: ...")

    # After Q&A exchange:
    store.store_qa("What is the top product?", "Widget Pro.")

    # Retrieve relevant context:
    results = store.search("anomalies")

    # Clear everything:
    store.clear()

Environment variables:
    CHROMA_DIR          = ./chroma_db   (storage path)
    OLLAMA_BASE_URL     = http://localhost:11434
    OLLAMA_EMBED_MODEL  = nomic-embed-text
"""

import logging
import os
from typing import ClassVar

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  CONTEXT STORE
# ═══════════════════════════════════════════════════════════════════════════


class ContextStore:
    """ChromaDB-backed memory for storing and retrieving analysis context.

    Collections:
        - "profiles"   : dataset profile outputs
        - "insights"   : analyst-generated insights
        - "reports"    : reporter outputs
        - "qa_history" : Q&A exchanges
    """

    # Collection names
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
        """
        Args:
            persist_dir: ChromaDB storage directory.
                         Defaults to CHROMA_DIR env var or ./chroma_db.
        """
        self.persist_dir = persist_dir or os.getenv("CHROMA_DIR", "./chroma_db")
        self._embeddings = OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    # ── Store methods ──────────────────────────────────────────────────────

    def store_profile(self, text: str) -> int:
        """Store a dataset profile.

        Returns:
            Number of chunks stored.
        """
        return self._store(text, collection=self.PROFILES)

    def store_insights(self, text: str) -> int:
        """Store analyst-generated insights.

        Returns:
            Number of chunks stored.
        """
        return self._store(text, collection=self.INSIGHTS)

    def store_report(self, text: str) -> int:
        """Store a reporter output.

        Returns:
            Number of chunks stored.
        """
        return self._store(text, collection=self.REPORTS)

    def store_qa(self, question: str, answer: str) -> int:
        """Store a Q&A exchange.

        Returns:
            Number of chunks stored.
        """
        text = f"Question: {question}\nAnswer: {answer}"
        return self._store(text, collection=self.QA_HISTORY)

    # ── Search methods ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        collection: str | None = None,
        k: int = 3,
    ) -> list[str]:
        """Search for relevant chunks in a specific collection.

        Args:
            query: Search query.
            collection: Collection to search. If None, searches all collections.
            k: Number of results per collection.

        Returns:
            List of relevant text chunks.
        """
        if collection:
            return self._search_collection(query, collection, k)

        # Search all collections
        results: list[str] = []
        for col in self.ALL_COLLECTIONS:
            results.extend(self._search_collection(query, col, k))
        return results

    def search_as_text(
        self,
        query: str,
        collection: str | None = None,
        k: int = 3,
    ) -> str:
        """Search and return results as a single joined string.

        Args:
            query: Search query.
            collection: Collection to search. If None, searches all.
            k: Number of results per collection.

        Returns:
            Concatenated text of relevant chunks, or empty string.
        """
        chunks = self.search(query, collection, k)
        if not chunks:
            return ""
        return "\n\n---\n\n".join(chunks)

    # ── Utility methods ────────────────────────────────────────────────────

    def clear(self, collection: str | None = None) -> None:
        """Delete all data from one or all collections.

        Args:
            collection: Collection to clear. If None, clears all.
        """
        import chromadb

        client = chromadb.PersistentClient(path=self.persist_dir)

        to_clear = [collection] if collection else self.ALL_COLLECTIONS

        for col_name in to_clear:
            try:
                client.delete_collection(col_name)
                logger.info(f"Cleared collection '{col_name}'")
            except Exception:
                pass

    def list_collections(self) -> list[str]:
        """List all existing collections.

        Returns:
            List of collection names that have data.
        """
        import chromadb

        client = chromadb.PersistentClient(path=self.persist_dir)
        return [c.name for c in client.list_collections()]

    # ── Private helpers ────────────────────────────────────────────────────

    def _store(self, text: str, collection: str) -> int:
        """Split text into chunks and store in a collection.

        Returns:
            Number of chunks stored.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text, nothing stored in '{collection}'")
            return 0

        chunks = self._splitter.split_text(text)

        Chroma.from_texts(
            texts=chunks,
            embedding=self._embeddings,
            collection_name=collection,
            persist_directory=self.persist_dir,
        )

        logger.info(f"Stored {len(chunks)} chunks in '{collection}'")
        return len(chunks)

    def _search_collection(self, query: str, collection: str, k: int) -> list[str]:
        """Search a single collection.

        Returns:
            List of matching text chunks.
        """
        try:
            vectorstore = Chroma(
                collection_name=collection,
                embedding_function=self._embeddings,
                persist_directory=self.persist_dir,
            )
            docs = vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]

        except Exception as e:
            logger.debug(f"Search in '{collection}' failed: {e}")
            return []
