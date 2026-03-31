# Memory & RAG Layer

AutoInsight AI uses a **ChromaDB-backed memory system** to persist analysis results across runs and enrich subsequent analyses with prior context. This is the Retrieval-Augmented Generation (RAG) layer.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LangGraph Pipeline                          │
│                                                                     │
│  profiler → analyst → visualizer → reporter_node → rag_storage_node│
│                                        ↑                   ↓       │
│                                   retrieves            stores      │
│                                        │                   │       │
│                              ┌─────────┴───────────────────┘       │
│                              │                                      │
│                    ┌─────────▼──────────────────────┐              │
│                    │    ChromaDB (./chroma_db)       │              │
│                    │  ┌──────────┐  ┌─────────────┐ │              │
│                    │  │ profiles │  │  insights   │ │              │
│                    │  ├──────────┤  ├─────────────┤ │              │
│                    │  │ reports  │  │ qa_history  │ │              │
│                    │  └──────────┘  └─────────────┘ │              │
│                    └────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principle:** `reporter_node` retrieves first, then `rag_storage_node` stores. This ordering ensures enrichment is always based on the previous run while the current run's fresh data is written afterwards.

---

## Architecture

### Two layers

| Layer | Class | Responsibility |
|-------|-------|----------------|
| Storage | `ContextStore` (`utils/memory.py`) | Raw ChromaDB operations — store, search, clear |
| Retrieval | `RAGAgent` (`agents/rag.py`) | Higher-level retrieval, formatting, deduplication, truncation |

`RAGAgent` owns a `ContextStore` instance and delegates all persistence to it. Agents in the pipeline interact with `RAGAgent`, never directly with `ContextStore`.

---

## Collections

ChromaDB holds four named collections, each storing a different type of artifact:

| Collection | Producer | Content |
|------------|----------|---------|
| `profiles` | Profiler agent | Dataset shape, types, null counts, descriptive statistics |
| `insights` | Analyst agent | Individual insights with priority and category metadata |
| `reports` | Reporter agent | Final markdown report |
| `qa_history` | Q&A interactions | Question/answer pairs |

### Chunk metadata

Every chunk stored in ChromaDB carries metadata used for filtered retrieval:

```python
# All collections
{"dataset_id": "sales_2024.csv", "source": "profiler"}

# Insights additionally include:
{"priority": "high", "category": "anomaly", "title": "Revenue spike in Q3"}
```

---

## Dataset isolation

Each uploaded file gets a `dataset_id` — all stored chunks are tagged with it, and all searches are scoped to it. This means:

- Analyses of `sales.csv` and `customers.csv` never bleed into each other.
- You can reset memory for a single dataset without touching others.

**How `dataset_id` is derived** (in `orchestration/graph.py`):

```python
dataset_id = ContextStore.make_dataset_id(file_name)  # returns filename.strip()
```

It is simply the filename, stripped of surrounding whitespace. The same file always produces the same `dataset_id`.

---

## The RAG enrichment loop

```
Run N:
  1. reporter_node  →  RAGAgent.get_analysis_context()  →  retrieves Run (N-1) chunks
  2. reporter uses retrieved context to enrich its report
  3. rag_storage_node  →  store.clear_dataset(dataset_id)  →  wiped Run (N-1) chunks
  4. rag_storage_node  →  stores Run N profile + insights + report

Run N+1:
  1. reporter_node retrieves Run N chunks  →  enriched with N's findings
  ...
```

**First run behaviour:** ChromaDB has no data for this `dataset_id`, so `get_analysis_context()` returns an empty string. The reporter produces its report without enrichment. The `rag_analysis_context` field in state remains empty, and the Memory tab shows a "no prior context" message.

**Subsequent runs:** The reporter finds context from the previous run and uses it to cross-reference, deepen, or validate findings before generating the current report.

---

## Replace-on-write semantics

`rag_storage_node` calls `store.clear_dataset(dataset_id)` **before** writing any new chunks. This enforces replace-on-write semantics:

- Each run stores exactly one generation of results per dataset.
- No accumulation of stale data across multiple runs.
- ChromaDB size stays bounded regardless of how many times the same file is re-analysed.

```python
# orchestration/graph.py — rag_storage_node
rag = RAGAgent(dataset_id=dataset_id)
rag.store.clear_dataset(dataset_id)   # ← drop previous run's chunks
rag.store.store_profile(...)           # ← write fresh chunks
rag.store.store_insights(...)
rag.store.store_report(...)
```

The clearing happens **after** `reporter_node` has already consumed the old context, so the enrichment path is unaffected.

---

## Cross-session persistence

ChromaDB persists to disk at `./chroma_db` (configurable via `CHROMA_DIR`). Data survives Streamlit restarts and process exits. This is intentional:

- Re-uploading the same file in a new session benefits from the previous session's analysis.
- The Memory tab and Q&A tab maintain continuity with a user's previous work.

Restarting Streamlit is not semantically meaningful from the user's perspective — the same files are still on disk, the same datasets are still in ChromaDB.

---

## Configuration

All settings read from environment variables via `config/settings.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL for embeddings |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |

---

## `RAGAgent` API reference

```python
from agents.rag import RAGAgent

rag = RAGAgent(dataset_id="sales_2024.csv")
```

### Retrieval

| Method | Use case |
|--------|----------|
| `get_analysis_context(k)` | Reporter enrichment — searches profiles + insights + reports |
| `get_context_for_query(query, k)` | Q&A — searches all collections |
| `get_context_for_followup(query, k)` | Follow-up questions — prioritises insights + Q&A history |
| `get_high_priority_insights(k)` | Executive summaries — only `priority="high"` insights |
| `get_insights_by_category(category, k)` | Category-specific views (trend, anomaly, correlation, …) |

All retrieval methods:
1. Query ChromaDB with semantic similarity search, scoped to `self.dataset_id`.
2. Deduplicate identical chunks.
3. Truncate to `MAX_CONTEXT_LENGTH` (3000 chars) at the last sentence boundary.

### Storage

| Method | What it stores |
|--------|---------------|
| `save_analysis(profile_text, insights, report_text)` | All pipeline outputs at once |
| `save_qa_exchange(question, answer)` | One Q&A turn |

### Management

| Method | Effect |
|--------|--------|
| `switch_dataset(dataset_id)` | Point the agent at a different dataset |
| `clear_memory()` | Delete **all** collections (full wipe) |
| `store.clear_dataset(dataset_id)` | Delete only chunks for one dataset |

---

## `ContextStore` storage internals

`_store()` splits text using `RecursiveCharacterTextSplitter` (chunk size 500, overlap 50) before writing to ChromaDB via `Chroma.from_texts()`. Each chunk gets a UUID, so the same text stored twice produces independent entries — hence the need for `clear_dataset()` before re-writing.

```
"1500 rows, 12 columns..."
        │
        ▼  RecursiveCharacterTextSplitter (500 / 50)
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   chunk 0     │  │   chunk 1     │  │   chunk 2     │
│ + metadata    │  │ + metadata    │  │ + metadata    │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        └──────────────────▼───────────────────┘
                  Chroma.from_texts()
                  ChromaDB collection
```

---

## Q&A and follow-up memory

The Q&A tab (not shown in the pipeline diagram above) integrates with the same memory layer:

1. `get_context_for_query(question)` retrieves relevant chunks before the LLM answers.
2. After the LLM answers, `save_qa_exchange(question, answer)` stores the turn in `qa_history`.
3. Subsequent questions can retrieve prior exchanges, making the Q&A session stateful.

---

## Limitations and future improvements

| Limitation | Current state | Improvement path |
|------------|---------------|------------------|
| `dataset_id` is just the filename | Two different files named `sales.csv` share a bucket | Use a content hash (SHA-256 of the file bytes) as `dataset_id` |
| No TTL / expiry | Old analyses accumulate indefinitely across many different datasets | Add a background job or on-startup prune for datasets not accessed in N days |
| No user-facing clear button | Power users cannot manually reset memory | Add a "Clear memory for this dataset" button to the Memory tab |
| Shared DB on disk | Multiple concurrent users would collide | Use a per-user subdirectory or switch to a managed vector DB (e.g. Qdrant, Weaviate) |
