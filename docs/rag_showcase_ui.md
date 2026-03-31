# RAG / Memory Showcase UI

This document describes the **Memory** tab and related UI enhancements that
make the RAG (Retrieval-Augmented Generation) system visible to users.

## Overview

AutoInsight-AI stores every analysis in a **ChromaDB vector database**,
partitioned by dataset ID.  On subsequent runs the pipeline automatically
retrieves relevant context and feeds it to downstream agents.  The Memory
tab surfaces this process so users can see—and interact with—the stored
knowledge.

## Architecture

```
  ┌────────────┐     build_memory_status()     ┌──────────────┐
  │ Pipeline   │ ─────────────────────────────▶ │ memory_view  │
  │ result     │                                │  (view-model)│
  └────────────┘                                └──────┬───────┘
                                                       │
                        ┌──────────────────────────────┘
                        ▼
              ┌──────────────────┐
              │  app/main.py     │   _render_memory_tab()
              │  (Streamlit UI)  │
              └──────────────────┘
```

* **`app/memory_view.py`** — view-model service.  Returns plain Python data
  classes; never imports Streamlit.
* **`app/main.py`** — renders the Memory tab using data from the view-model.
* **`diagnostics/renderer.py`** — enhanced Memory & RAG section with status
  badges and event counts.

## Memory Tab Sections

### 1. Memory Status Overview
Four metrics: Dataset ID, memory availability, prior context size, and
whether artifacts were stored in the current run.

### 2. Retrieved Context Preview
When prior context was found and injected into the Reporter agent, users
can expand it here.  On the first run this section explains that no prior
context exists yet.

### 3. Stored Artifact Summary
Shows which artifact types (profile, insights, report) were persisted, the
total number of chunks, and collections used.

### 4. Interactive Memory Retrieval Demo
Three sub-tabs let users query ChromaDB live:

| Sub-tab | Function called | Description |
|---------|----------------|-------------|
| 🔝 High Priority | `retrieve_high_priority_insights()` | Top-5 high-priority insights |
| 🏷️ By Category | `retrieve_insights_by_category()` | Filter by trend / anomaly / correlation / distribution / general |
| 📑 Full Context | `retrieve_analysis_context()` | Combined profiles + insights + reports |

These are the **same functions** that future agents (Text-to-Code,
conversational Q&A) will use.

### 5. Why Memory Matters
A brief explainer that educates users about cross-run enrichment and the
value of the vector store.

## Report Tab Enhancement

When `rag_analysis_context` is non-empty the Report tab displays an info
banner:

> 📚 This report was **enriched with context** retrieved from previous
> analyses stored in memory.

This makes the RAG contribution visible without requiring users to
navigate to the Memory tab.

## First Run vs. Second Run

| Aspect | First run | Second run |
|--------|-----------|------------|
| Prior context | None (empty) | Retrieved from ChromaDB |
| Report banner | Not shown | "Enriched with memory" |
| Memory tab – context preview | Placeholder message | Actual context text |
| Memory tab – retrieval demo | Empty results | Live results |

## Diagnostics Enhancement

The Memory & RAG section in the Diagnostics tab now shows:

* **Event summary line** — e.g. "3/3 events succeeded"
* **Status badges** — ✅ / ❌ / ⏭️ / ⬜ inline with each event pill
* Consistent color-coded styling (green / red / amber / slate)

## Graceful Degradation

* If ChromaDB is unreachable, `build_memory_status()` sets
  `memory_available = False` and the interactive demo section is hidden.
* All retrieval helpers return `RetrievalResult(empty=True)` on any
  exception — no crashes, no mandatory features.
* The Memory tab still renders status and stored artifacts even when
  live queries are unavailable.

## Testing

Run the memory-view tests:

```bash
python -m pytest tests/test_memory_view.py -v
```

Coverage includes:
- `build_memory_status` with empty, full, and failed pipeline states
- Retrieval helpers with mocked RAGAgent (success + failure paths)
- Graceful degradation when imports fail
- Frozen dataclass invariants
