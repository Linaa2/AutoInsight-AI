# Streamlit UI

This document describes the product layer exposed in `app/main.py`.

## Product Goal

The Streamlit app is the primary interface for the project. It lets a user:

- upload a tabular dataset
- inspect a quick preview
- launch the full multi-agent pipeline
- watch the pipeline progress live
- explore the final outputs through dedicated tabs

## Main User Journey

```text
Open app
  -> Upload dataset
  -> Preview dataset KPIs and sample rows
  -> Run full analysis
  -> Watch pipeline status update
  -> Explore final tabs
```

## Input Support

Supported upload formats:

- CSV
- Excel (`.xlsx`, `.xls`)
- Parquet

File loading is handled by `tools/data_loader.py`.

## Dataset Preview

Before the pipeline runs, the app computes a deterministic preview using `DataProfiler`.

Preview elements:

- row count
- column count
- duplicate count
- missing percentage
- memory usage
- preview of the first rows

## Pipeline Status

The app shows the pipeline status before, during, and after execution.

### During streaming

Live status is rendered as plain HTML to avoid Streamlit reruns that would interrupt the generator.

### After completion

The final status diagram is rendered through `streamlit_flow`, which wraps React Flow.

### Canonical display order

```text
START -> Profiler -> Analyst -> Critic -> Uncertainty
         -> Visualizer -> Reporter -> Memory -> END
```

The diagram is intentionally split across two rows for readability.

## Final Result Tabs

After a completed run, the app dynamically assembles tabs based on available outputs.

### Core tabs

| Tab | Purpose |
|---|---|
| `Profile` | Render the profiler markdown |
| `Insights` | Show structured insights as cards or markdown |
| `Evaluation` | Show critique and confidence information |
| `Visualizations` | Render generated Plotly charts and chart metadata |
| `Report` | Display the final markdown report |

### Support tabs

| Tab | Purpose |
|---|---|
| `Memory` | Show memory status and retrieval demos |
| `Observability` | Show LangFuse and run-observability summary |
| `Diagnostics` | Show node-level execution telemetry and final pipeline diagram |
| `LLM Judge` | Run post-hoc quality evaluation |

## Progressive Rendering

During `stream_analysis()`:

- the app updates the live status area after each node
- partial content is shown in progressively unlocked tabs
- completed nodes show their actual content
- future nodes show waiting or running placeholders

This keeps the UI responsive even when model calls are slow.

## Session State

The app relies on Streamlit session state for run continuity.

Important keys:

- `analysis_result`
- `last_file_name`
- `analysis_running`
- `analysis_cancelled`
- `pipeline_eval`
- `dataset_id`

Behavior:

- uploading a new file resets the previous result
- re-running the same file preserves the same derived `dataset_id`
- post-run evaluation is stored separately from the main pipeline result

## Memory UX

When a run is successfully stored in ChromaDB:

- the sidebar shows a memory success indicator
- the Memory tab becomes available
- re-running the same file can enrich the report with prior context

## Observability UX

When LangFuse is enabled and configured:

- the sidebar shows LangFuse status
- the current run can expose a trace id
- the Observability tab links the run to LangFuse concepts

Even when LangFuse is disabled, local diagnostics remain available.

## UI Boundaries

The Streamlit layer intentionally avoids owning business logic.

Responsibilities that stay outside `app/`:

- graph wiring
- prompt construction
- model routing
- deterministic profiling
- memory persistence
- LangFuse client integration

The app should render results and orchestrate user actions, not implement analysis logic directly.
