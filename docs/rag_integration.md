# RAG Integration — ChromaDB Memory for AutoInsight AI

## Overview

**Phase 5** integrates persistent, per-dataset memory into the AutoInsight AI pipeline using
**ChromaDB** (local vector store) and **Ollama embeddings** (`nomic-embed-text` by default).

The goal is two-fold:

1. **Store** each analysis run so future runs on the same dataset can reference prior context.
2. **Retrieve** and **inject** that prior context into the Reporter agent, enabling richer,
   historically-aware reports without any user effort.

---

## Architecture

```
File upload
    │
    ▼
ContextStore.make_dataset_id(filename)   ← deterministic, whitespace-stripped
    │
    ▼
_make_initial_state(df, file_name, dataset_id)
    │   └─ initialises PipelineState with dataset_id + graph_trace=[]
    │
    ▼
LangGraph pipeline:
  profiler ──► analyst ──► visualizer ──► reporter ──► rag_storage ──► END
                                              │               │
                                  retrieves prior context       stores outputs
                                  on demand (best-effort)       (current run)
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| `rag_storage_node` runs **after** reporter | Storage is isolated from the analysis agents and remains best-effort |
| RAG is **always optional** — every node degrades gracefully | No Ollama / ChromaDB = identical behaviour |
| `dataset_id` is the **filename** (stripped) | Simple, human-readable, stable across sessions |
| `rag_storage_node` is **not** in `_AGENTS_ORDER` | It's infrastructure, not a user-visible analysis step |
| Retrieval happens inside `reporter_node` | Avoids adding startup latency before the first streamed result |

---

## Components

### `config/settings.py`

Two new settings added (both overridable via environment variables):

```python
OLLAMA_EMBED_MODEL: str  # default "nomic-embed-text"
CHROMA_DIR: str          # default "./chroma_db"
```

### `utils/memory.py` — `ContextStore`

Low-level ChromaDB abstraction.  All configuration is read from the centralised
`settings` singleton (no raw `os.getenv` calls).

```python
from utils.memory import ContextStore

store = ContextStore()                                     # uses settings.CHROMA_DIR
store.store_profile("Dataset has 1500 rows...", dataset_id="sales.csv")
store.store_insights([...], dataset_id="sales.csv")
store.store_report("## Executive Summary...", dataset_id="sales.csv")

results: list[str] = store.search("revenue anomalies", dataset_id="sales.csv")
```

**`make_dataset_id(filename: str) → str`** — the canonical ID generator:
strips leading/trailing whitespace and returns the filename as-is.  All IDs
in the system are derived from this single method.

**Collections:**

| Collection | Source agent | Stored content |
|---|---|---|
| `profiles` | Profiler | Profile markdown (stats + LLM description) |
| `insights` | Analyst | Structured insight dicts (one doc per insight) |
| `reports` | Reporter | Full report markdown |
| `qa_history` | Text-to-Code / Q&A (future) | Q&A exchanges |

### `agents/rag.py` — `RAGAgent`

High-level retrieval/storage service bound to a `dataset_id`.  Wraps `ContextStore`
with convenience methods and deduplication + truncation logic.

```python
from agents.rag import RAGAgent

rag = RAGAgent(dataset_id="sales.csv")

# After pipeline completes:
rag.save_analysis(profile_text="...", insights=[...], report_text="...")

# Before Reporter (via _make_initial_state):
context = rag.get_analysis_context()   # empty on first run, populated on subsequent runs

# For Q&A / Text-to-Code:
context = rag.get_context_for_query("top performing product")
high_priority = rag.get_high_priority_insights(k=5)
```

`MAX_CONTEXT_LENGTH = 3000` — all retrieved context is deduplicated and truncated
before injection.

### `orchestration/state.py` — New Fields

```python
class MemoryTraceEntry(TypedDict, total=False):
    event: str       # "store_profile" | "store_insights" | "store_report" | "retrieve_context"
    status: str      # "success" | "skipped" | "empty" | "failed"
    dataset_id: str
    collection: str | None
    chunks: int      # documents stored / retrieved
    message: str

# PipelineState additions:
dataset_id: str              # stable per-file identifier
rag_analysis_context: str    # prior-run context for reporter (empty on first run)
rag_stored: bool             # True when rag_storage_node ran successfully
rag_summary: str             # e.g. "Stored profile, insights, report (12 chunks)"
memory_trace: list[MemoryTraceEntry]
```

### `orchestration/graph.py` — `rag_storage_node` + `_make_initial_state`

**`_make_initial_state(df, file_name, dataset_id=None) → PipelineState`**

Computes `dataset_id`, registers the DataFrame in the in-process registry, and
starts the LangFuse trace. It returns a valid initial state with
`rag_analysis_context=""`; prior context retrieval is deferred to `reporter_node`.

**`rag_storage_node(state) → PipelineState`**

Best-effort storage node that:

1. Returns early (status=`skipped`) if there are no analysis outputs to store.
2. Instantiates `RAGAgent` and stores profile, insights, and report individually.
3. Each storage operation is wrapped in its own `try/except` — a failure on one
   does not prevent the others from running.
4. Appends one `MemoryTraceEntry` per operation (success or failure).
5. The outer `try/except` catches any crash during instantiation — the node
   **never propagates exceptions** to the graph.

### `agents/reporter.py` + `orchestration/graph.py` — RAG Context Injection

`ReporterAgent.run()` accepts an optional `rag_context: str = ""` parameter.
When non-empty, the context is appended to the human prompt as a supplementary section:

```
## 📚 Context from Previous Analyses
The following is relevant context retrieved from a previous run on this dataset.
Use it only if it adds value to the current report.

<retrieved context>
```

`reporter_node` retrieves the prior context just-in-time, then passes it into
`ReporterAgent.run(...)`:

```python
result = agent.run(
    profiler_output=profile_md or "",
    analyst_output=insights_md or "",
    insights=insights,
    visualizer_output=viz_result,
    rag_context=state.get("rag_analysis_context") or "",   # ← RAG injection
)
```

### `diagnostics/renderer.py` — Memory & RAG Section

The Diagnostics tab now includes a **Section 4: Memory & RAG** that shows:

- Dataset ID badge
- Overall `✅ Stored` / `⬜ Not stored` status
- Human-readable `rag_summary` caption
- Per-event pills (color-coded green/red/yellow based on success/failed/skipped)

```
🧠 Memory & RAG
Dataset ID: sales.csv
✅ Stored — "Stored profile, insights, report (12 chunks) for dataset 'sales.csv'"

📊 Store profile  → profiles        Profile markdown stored
💡 Store insights → insights  · 8 chunks  Stored 3 structured insights
📄 Store report   → reports   · 4 chunks  Report markdown stored
```

### `app/main.py` — Dataset ID Lifecycle

`dataset_id` is computed once per uploaded file and stored in `st.session_state`:

```python
# On new file upload:
st.session_state["dataset_id"] = ContextStore.make_dataset_id(uploaded_file.name)

# Passed to stream_analysis:
stream_analysis(df, file_name=file_name, dataset_id=dataset_id or None)
```

After a successful analysis, the sidebar shows a `🧠 Analysis stored in memory` badge
with the `dataset_id` caption.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |

Set via `.env` or shell export:

```bash
export OLLAMA_EMBED_MODEL=mxbai-embed-large
export CHROMA_DIR=/data/autoinsight-chroma
```

---

## Running with RAG Enabled

1. **Start Ollama** and pull the embedding model:

   ```bash
   ollama serve
   ollama pull nomic-embed-text
   ```

2. **Launch the app** — ChromaDB is created automatically:

   ```bash
   make run
   # or
   uv run streamlit run app/main.py
   ```

3. **Upload a dataset and run analysis.** After the pipeline completes:
   - The sidebar shows `🧠 Analysis stored in memory`.
   - Run again with the **same file** — the Reporter will now receive prior context.

---

## Running without RAG (Degraded Mode)

If Ollama is not running or ChromaDB is unavailable, **the app still works normally**.
All RAG operations fail silently:

- `rag_analysis_context` stays `""` → reporter behaves as before.
- `rag_storage_node` records `status="failed"` in `memory_trace` but never raises.
- `rag_stored=False` → sidebar badge is not shown.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Run 1 (first upload)                        │
│                                                                 │
│  profiler → analyst → visualizer → reporter → rag_storage      │
│                                        │            │           │
│                              no prior context   stores:         │
│                              (rag_ctx = "")     - profile       │
│                                                 - insights      │
│                                                 - report        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Run 2 (same file, re-run)                   │
│                                                                 │
│  _make_initial_state ──► retrieves prior context from ChromaDB  │
│                                  │                              │
│  profiler → analyst → visualizer → reporter ─────────────────► │
│                                      │          rag_ctx injected│
│                                      ▼         into prompt      │
│                               report enriched                   │
│                               with prior run context            │
│                                            ↓                    │
│                                    rag_storage (overwrites)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Testing

All RAG tests are in [tests/test_rag.py](../tests/test_rag.py) and run without
Ollama or ChromaDB (mocked or skipped via early-return paths):

```bash
uv run pytest tests/test_rag.py -v
```

**Test coverage:**

| Test | What it verifies |
|---|---|
| `test_make_dataset_id_*` | ID is stable, strips whitespace, unique per file |
| `test_format_and_truncate_*` | Deduplication, truncation, sentence-boundary cut |
| `test_rag_storage_node_skipped_*` | Empty state → `rag_stored=False`, trace=`skipped` |
| `test_rag_storage_node_graceful_*` | ChromaDB crash → no exception propagation |
| `test_make_initial_state_*` | dataset_id derived/explicit, rag_context pre-loaded |
| `test_reporter_agent_*` | Backward compat, RAG context injection, error handling |

---

## Future Extensions

- **Text-to-Code Q&A:** `RAGAgent.get_context_for_query(user_question)` returns relevant
  context for the Text-to-Code agent's code generation prompt.
- **Cross-dataset search:** pass `dataset_id=None` to `ContextStore.search()` to search
  across all uploaded files.
- **Multi-turn Q&A history:** `RAGAgent.save_qa_exchange(question, answer)` stores each
  exchange for follow-up context.
- **Insight prioritisation:** `RAGAgent.get_high_priority_insights()` and
  `get_insights_by_category(category)` are available for targeted retrieval.
