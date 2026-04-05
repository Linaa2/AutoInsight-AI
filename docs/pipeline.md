# Pipeline

This document explains the automated analysis pipeline executed by LangGraph.

## Automated Workflow

The current graph order is:

```text
START
  -> profiler
  -> analyst
  -> critic
  -> uncertainty
  -> visualizer
  -> reporter
  -> rag_storage
END
```

The pipeline is intentionally linear. Each node reads the shared state, performs its work, writes new keys, and appends a trace entry.

## Execution Semantics

Each node follows the same pattern:

1. read only the state keys it needs
2. call the corresponding agent or deterministic tool
3. write only its own outputs
4. append one `NodeTraceEntry` to `graph_trace`
5. fail gracefully when possible

## Node-by-Node Breakdown

| Node | Inputs | Outputs | Notes |
|---|---|---|---|
| `profiler` | `df_ref` or `df_dict` | `profile_data`, `profile_markdown` | Deterministic profiling plus narrative interpretation |
| `analyst` | `profile_markdown`, `profile_data`, dataset sample | `insights`, `insights_markdown` | Generates structured insights |
| `critic` | `insights`, `profile_data` | `critiques`, `critic_output` | Reviews insights in batches |
| `uncertainty` | `insights`, `critiques`, `profile_data` | `confidence_scores`, `uncertainty_output` | Hybrid rule-based and LLM-based confidence scoring |
| `visualizer` | dataset, profile context, insight context | `visualization_result` | Generates chart specs and executes chart code |
| `reporter` | profile, insights, critique, confidence, charts, optional RAG context | `report_markdown`, optionally `rag_analysis_context`, `memory_trace` | Final synthesis step |
| `rag_storage` | profile, insights, report, `dataset_id` | `rag_stored`, `rag_summary`, `memory_trace` | Best-effort persistence to ChromaDB |

## State Flow

### 1. Initial state

Before graph execution:

- the DataFrame is registered in-process as `df_ref`
- `dataset_id` is derived from the uploaded filename if not provided
- `langfuse_trace_id` is created when LangFuse is enabled
- `rag_analysis_context` starts empty

### 2. Analysis state

As the pipeline progresses:

- deterministic outputs are added first
- LLM outputs are layered on top of deterministic context
- diagnostics and memory traces grow cumulatively

### 3. Final state

A successful run typically includes:

- dataset identity
- profile markdown and profile data
- structured insights and markdown insights
- critiques and confidence scores
- visualization result
- final report
- memory trace and storage summary
- node trace and optional LangFuse trace id

## Streaming vs Final Run

There are two public execution entry points:

### `run_analysis()`

- runs the full graph synchronously
- returns the terminal `PipelineState`

### `stream_analysis()`

- streams node updates after each completed node
- is used by the Streamlit app for progressive rendering
- injects stable metadata such as `file_name`, `dataset_id`, and `langfuse_trace_id`

## Error Handling Philosophy

The pipeline is designed to degrade gracefully.

### Main analysis nodes

If a node fails:

- the node records a failed trace entry
- an error string is written into the state
- downstream rendering can still show partial results where possible

### Memory layer

`rag_storage` is explicitly best-effort:

- it must never crash the whole run
- storage failures are recorded in `memory_trace`
- the report is already complete before storage starts

### LangFuse

LangFuse is optional:

- invalid or missing credentials should disable tracing for the run
- LangFuse failures should not block the analysis pipeline

## Pipeline Status in the UI

The Streamlit app represents the pipeline in two ways:

- live in-progress HTML status during streaming
- final two-row `streamlit_flow` diagram after the run completes

Canonical status order:

```text
START -> Profiler -> Analyst -> Critic -> Uncertainty
         -> Visualizer -> Reporter -> Memory -> END
```

The `LLM Judge` appears after `END` in the UI because it is a post-run evaluation feature, not part of the automated LangGraph pipeline.

## What Is Not in the Automated Pipeline

These features are outside the main graph:

- `LLM Judge` post-run evaluation
- interactive memory retrieval demo in the UI
- LangFuse trace viewing in the browser

They consume results from the completed run but do not alter the core workflow.
