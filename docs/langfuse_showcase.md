# LangFuse Observability Showcase

This document explains how LangFuse is integrated into AutoInsight-AI,
how the Observability tab works, how to demonstrate its value, and how future
agents plug into the same framework.

---

## 1. Implementation Logic

### Integration architecture

```
  ┌─────────────────────────────────────────────────────┐
  │  LangGraph Pipeline                                 │
  │                                                     │
  │  _make_initial_state()                              │
  │    └── lf_monitor.begin_run()  → trace_id           │
  │                                                     │
  │  profiler_node()                                    │
  │    ├── lf_monitor.get_llm_callbacks()               │
  │    └── lf_monitor.node_span("profiler")  ← span     │
  │                                                     │
  │  analyst_node()  / visualizer_node() / reporter_node│
  │    └── same pattern (callbacks + node_span)         │
  │                                                     │
  │  rag_storage_node()                                 │
  │    └── lf_monitor.log_event("rag-storage")          │
  │                                                     │
  │  _make_initial_state() preload                      │
  │    └── lf_monitor.log_event("rag-context-retrieval")│
  └─────────────────────────────────────────────────────┘
              │ langfuse_trace_id written to PipelineState
              ▼
  ┌──────────────────────┐     ┌──────────────────────────┐
  │ app/langfuse_view.py │     │ utils/langfuse_client.py  │
  │ (view-model layer)   │     │ (singleton monitor)       │
  │                      │     │                          │
  │ build_langfuse_       │     │ LangFuseMonitor          │
  │   run_summary(result)│     │ ├── begin_run()           │
  │   → LangfuseRunSummary     │ ├── node_span()           │
  │                      │     │ ├── get_llm_callbacks()   │
  │ build_trace_url()    │     │ ├── log_event()           │
  └──────────────────────┘     │ └── flush()              │
              │                └──────────────────────────┘
              ▼
  ┌──────────────────────┐
  │ app/main.py          │
  │ _render_observability │
  │ _tab()               │
  │                      │
  │ → 7-section UI panel │
  └──────────────────────┘
```

### Where trace IDs come from

1. `_make_initial_state()` calls `lf_monitor.begin_run(dataset_id, file_name, run_id)`.
2. The monitor creates a trace ID via `langfuse.create_trace_id()` and stores it internally.
3. The trace ID is written into `PipelineState["langfuse_trace_id"]`.
4. After the full run, `app/langfuse_view.py` reads `result["langfuse_trace_id"]` and
   constructs the trace URL as `{LANGFUSE_BASE_URL}/trace/{trace_id}`.

### How the UI gets observability data

The Streamlit app never calls the LangFuse SDK directly.
It calls `build_langfuse_run_summary(result)` which:
- reads `langfuse_trace_id` from the pipeline result
- reads `graph_trace` to derive per-node summaries
- reads `memory_trace`, `rag_stored`, `rag_analysis_context` for RAG info
- checks `is_langfuse_enabled()` and `settings.LANGFUSE_BASE_URL`
- returns a frozen `LangfuseRunSummary` dataclass

This keeps the UI contract stable: future changes to LangFuse internals
only require updating `langfuse_client.py` and `langfuse_view.py`.

### How this relates to diagnostics and logger

| Layer | Purpose | Audience |
|---|---|---|
| `logging` (Python logger) | Operational log — local debug, container stdout | Developer |
| `diagnostics/renderer.py` | Structured in-app telemetry (CPU, RAM, tokens) | Developer / Power user |
| `app/langfuse_view.py` → Observability tab | External GenAI observability — prompts, runs, spans | Teacher / Recruiter / Developer |
| LangFuse UI | Full prompt/response, run history, cross-run comparison | Developer / MLOps |

They complement rather than duplicate each other. Local diagnostics cover
resource usage. LangFuse covers prompt quality, response tracing, and history.

---

## 2. How to Demonstrate LangFuse Usefulness

### Demo flow — First run

1. Start the stack: `make langfuse-up && make run`
2. Upload `data/sample_sales.csv`
3. Click **Run Full Analysis**
4. After completion, open the **🔭 Observability** tab
5. Show the class:
   - The LangFuse Status card (enabled, host, trace ID)
   - The Run Summary metrics (duration, node counts)
   - The **"Open this run in LangFuse"** button
6. Click the button → opens LangFuse UI directly at the trace
7. In LangFuse, show:
   - The top-level trace (one per pipeline run)
   - Child spans: `profiler`, `analyst`, `visualizer`, `reporter`, `rag-storage`
   - Click any span → show the full prompt text and model response
   - Show `rag-context-retrieval` span → no prior context on first run

**Key message:** *"One analysis run = one observable trace. Every agent call is a span. We can inspect every prompt without touching the code."*

### Demo flow — Second run (memory enrichment)

1. Click **Re-run Analysis** (same file)
2. Open the **🔭 Observability** tab again
3. Point out:
   - A *new* trace was generated (different trace ID)
   - **Prior Context: Found ✅** in the RAG / Memory section
   - The info banner: *"Prior analysis context was retrieved …"*
4. Open LangFuse — show the **second** trace:
   - `rag-context-retrieval` span now has non-empty output
   - The reporter span has additional context in its prompt input
5. Compare the two traces side-by-side in LangFuse

**Key message:** *"LangFuse lets us verify that RAG retrieval is actually working — we can see exactly what context was retrieved and how it affected the model's prompt."*

### Demo flow — Failure / debug case

1. Stop Ollama (`ollama stop`) or point to an invalid model
2. Run the analysis
3. In the **🔭 Observability** tab, show:
   - Status: `Failed` or `Partial`
   - Node table: the failing node has ❌ and an error message
4. Open LangFuse — the failed span shows the exact error and timing
5. Restart Ollama and re-run → green trace

**Key message:** *"When something fails, a developer doesn't need to search logs — they go to LangFuse, click the failed span, see the error message and the full context."*

### What to click in Streamlit

| Tab | What to show |
|---|---|
| 🔭 Observability | LangFuse status, trace link, node table, RAG section |
| 🧠 Memory | Stored artifacts, retrieved context preview |
| 🔧 Diagnostics | Per-node resource usage, React Flow diagram |

The Observability tab is the *story* tab — this is what you show to a
teacher or recruiter who asks "how do you make this observable?"

---

## 3. UI Design Explanation

### Why 7 sections?

| Section | Purpose |
|---|---|
| 1. LangFuse Status Card | One glance — is this enabled and working? |
| 2. Run Summary | Aggregated pipeline outcome — pass/fail/timing |
| 3. Open in LangFuse | The most important affordance — jump to external trace |
| 4. Node Execution Table | Show that LangFuse tracks the *full pipeline*, not one LLM call |
| 5. RAG / Memory Observability | Connect the memory story to observability |
| 6. Why LangFuse Matters | Educate — table comparing local diagnostics vs LangFuse |
| 7. Raw Identifiers Expander | Developer utility — copy trace IDs, inspect raw values |

### Why not show raw prompts in the app?

Raw prompts are large (hundreds of tokens), change per run, and showing
them in Streamlit would:
1. Clutter the product experience for non-developer users
2. Duplicate what LangFuse already does (better)
3. Create a temptation to hard-code prompt display logic

The Streamlit app *summarises* and *links out*. LangFuse is the place to
read actual prompts. This is the correct architectural boundary.

### Why LangFuse links matter

A clickable `"Open in LangFuse"` button transforms LangFuse from a backend
logging system into an integrated product feature. Users can move
seamlessly from analysing a result to inspecting the underlying trace.
This is equivalent to the "Inspect" button in a browser — the expert
path that does not interrupt the main experience.

---

## 4. Future Extensibility

### How new agents appear automatically

Node summaries in the Observability tab are derived from `graph_trace`, not
from a hardcoded list. Any new graph node (Text-to-Code, Critic, QA …) will:

1. Appear automatically in the **Node Execution Table** as soon as it adds
   a `NodeTraceEntry` to `graph_trace`.
2. Get its own LangFuse span if it calls `lf_monitor.node_span(node_name)`.
3. Display model info if it populates `NodeTelemetry.model_info` via
   `build_telemetry()`.

The canonical node ordering in `_OBS_NODE_ORDER` in `app/main.py` controls
display order — add new nodes there when ready.

### Plugging a new agent into the observability model

```python
# In the new node function (e.g. text_to_code_node):

def text_to_code_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    res_before = collect_resource_snapshot()
    callbacks = lf_monitor.get_llm_callbacks()              # ← LangChain LLM tracing

    with lf_monitor.node_span("text_to_code", metadata={"dataset_id": ...}):  # ← LangFuse span
        # ... agent logic ...
        llm_start = time.time()
        output = agent.run(context=..., callbacks=callbacks)  # ← callbacks enable prompt tracing
        llm_end = time.time()

    telem = build_telemetry(task="code", llm_start=llm_start, llm_end=llm_end, ...)
    entry = _trace_entry("text_to_code", status="success", ...)        # ← feeds graph_trace
    return {..., "graph_trace": _append_trace(state, entry)}
```

No changes to `app/langfuse_view.py` or `_render_observability_tab()` are needed.
The node will appear in the UI automatically.

### Plugging into RAG observability

For agents that perform RAG retrieval, add a `log_event` call:

```python
lf_monitor.log_event("rag-retrieval",
    input={"query": ..., "dataset_id": ...},
    output={"retrieved": True, "context_chars": len(ctx)})
```

This event appears as a span in the LangFuse trace and is counted in the
`memory_events_count` field of `LangfuseRunSummary` if you also add a
corresponding `MemoryTraceEntry` to `memory_trace`.

---

## 5. Configuration Reference

| Variable | Where set | Effect |
|---|---|---|
| `LANGFUSE_ENABLED` | `.env` | Master switch — `true` enables all tracing |
| `LANGFUSE_PUBLIC_KEY` | `.env` (via `setup_langfuse.py`) | Auth public key |
| `LANGFUSE_SECRET_KEY` | `.env` (via `setup_langfuse.py`) | Auth secret key |
| `LANGFUSE_BASE_URL` | `.env` | LangFuse server URL (default: `http://localhost:3001`) |

Run `make langfuse-up` to start a local LangFuse instance and auto-populate
these variables. The trace link in the Observability tab uses `LANGFUSE_BASE_URL`
to construct the URL.
