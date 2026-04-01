# Memory and RAG

This document describes the ChromaDB-backed memory layer used by AutoInsight-AI.

## Purpose

The memory subsystem allows the application to reuse context from previous runs on the same dataset.

This supports:

- better continuity across repeated analyses
- richer final reports
- future follow-up workflows such as Q&A or text-to-code assistance

## Main Components

| Component | Role |
|---|---|
| `utils/memory.py` | ChromaDB storage and retrieval implementation |
| `agents/rag.py` | Retrieval service facade used by other layers |
| `orchestration/graph.py` | Retrieval before report generation and storage after reporting |
| `app/memory_view.py` | UI-facing memory view model |

## Storage Model

The memory layer stores content in multiple logical collections:

- `profiles`
- `insights`
- `reports`
- `qa_history`

Each stored chunk includes metadata such as:

- `dataset_id`
- source agent
- priority for insights
- category for insights

## Dataset Isolation

All retrieval is scoped by `dataset_id`.

This prevents context from unrelated datasets from leaking into a run.

`dataset_id` is typically derived from the uploaded filename through `ContextStore.make_dataset_id()`.

## Retrieval Flow

The current retrieval pattern is:

1. the pipeline runs normally through profiler, analyst, critic, uncertainty, and visualizer
2. just before report generation, the reporter step checks whether prior context is needed
3. if no context is already present, `RAGAgent.get_analysis_context()` retrieves relevant prior chunks
4. that context is injected into the reporter prompt

This means retrieval enriches the report without blocking earlier steps unnecessarily.

## Storage Flow

After the report is written:

1. `rag_storage` clears the previous chunks for the same dataset
2. the current profile, insights, and report are stored
3. storage activity is written into `memory_trace`

This is replace-on-write behavior, not append-forever behavior.

## Why Replace-on-Write

Without replacement, re-running the same dataset would accumulate stale chunks and pollute future retrieval.

Replace-on-write ensures:

- one clean memory snapshot per dataset
- predictable retrieval behavior
- bounded storage growth for repeated analyses

## Retrieval APIs

The `RAGAgent` exposes focused retrieval helpers:

- `get_analysis_context()`
- `get_context_for_query()`
- `get_context_for_followup()`
- `get_high_priority_insights()`
- `get_insights_by_category()`

It also exposes storage helpers for:

- full analysis persistence
- Q&A storage
- memory clearing

## UI Integration

The Streamlit app includes a dedicated `Memory` tab after a completed run.

That tab can show:

- whether memory is available
- whether prior context was found
- what was stored during the current run
- interactive retrieval demos by category or priority

The sidebar also shows when a completed analysis has been stored successfully.

## Degraded Mode

The application is designed to continue working even if the memory layer is unavailable.

If ChromaDB or embeddings fail:

- the main report can still be generated
- `rag_analysis_context` stays empty
- `rag_storage` records failure in `memory_trace`
- the pipeline should not crash solely because memory is unavailable

## Embeddings

The memory subsystem currently uses Ollama embeddings.

Default embedding model:

```text
nomic-embed-text
```

Configured by:

```text
OLLAMA_EMBED_MODEL
```

## Operational Notes

- first run on a dataset typically has no prior context
- second run on the same dataset can surface prior context in the reporter
- memory is local by default because ChromaDB persists under `CHROMA_DIR`

## Best Mental Model

Think of memory as a reusable analysis cache:

- retrieval happens before the final narrative is written
- storage happens after the current run is complete
- everything is scoped to one dataset identity
