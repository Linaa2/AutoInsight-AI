# Architecture

This document describes the current system architecture of AutoInsight-AI.

## System Overview

AutoInsight-AI is a multi-agent data analysis application with four major layers:

```text
User Upload
  -> Streamlit product layer
  -> LangGraph orchestration layer
  -> Agent and utility layer
  -> Storage / observability layer
```

The system combines deterministic analytics, LLM-based reasoning, chart generation, persistent memory, and optional observability.

## Core Design Principles

- One canonical orchestration layer in `orchestration/`
- One shared typed state contract in `PipelineState`
- Agents remain specialized and loosely coupled
- Deterministic computation stays outside the LLM when possible
- Observability and memory are best-effort and should not break the main product flow

## Main Runtime Components

| Layer | Main modules | Role |
|---|---|---|
| Product UI | `app/main.py`, `app/*_view.py` | Upload, progressive rendering, final result tabs |
| Orchestration | `orchestration/graph.py`, `orchestration/state.py` | Workflow execution, state passing, node tracing |
| Agents | `agents/*.py` | LLM-driven reasoning and synthesis |
| Deterministic tools | `tools/*.py`, `visualization/*.py` | Profiling, data loading, chart execution/parsing |
| Shared services | `utils/llm.py`, `utils/prompt_loader.py`, `utils/memory.py`, `utils/langfuse_client.py` | LLM client factory, prompt access, ChromaDB memory, LangFuse tracing |
| Evaluation | `evaluation/*.py` | LLM-as-Judge and score schemas |

## Repository Structure

```text
agents/           Agent implementations
app/              Streamlit product layer
config/           Settings and shared prompts
data/             Demo datasets
diagnostics/      Pipeline diagnostics and status diagrams
evaluation/       Offline and post-run evaluation logic
orchestration/    Canonical LangGraph graph and state
tools/            Deterministic data utilities
utils/            Shared infrastructure helpers
visualization/    Visualization parsing and chart execution
```

## Canonical Orchestration

The authoritative graph is defined in `orchestration/graph.py`.

Current automated pipeline:

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

Important distinction:

- `critic` and `uncertainty` are part of the automated pipeline
- `LLM Judge` is not part of the automated LangGraph pipeline
- `rag_storage` is infrastructure, not a user-facing analysis agent

## Shared State Contract

All graph nodes exchange data through `PipelineState` in `orchestration/state.py`.

Key design choices:

- `TypedDict(total=False)` so each node returns only the keys it writes
- `df_ref` stores the normal in-process DataFrame handle
- `df_dict` remains as a backward-compatible JSON-safe fallback
- `graph_trace` accumulates structured execution information for the UI

Major state groups:

| Group | Example keys |
|---|---|
| Input identity | `df_ref`, `file_name`, `dataset_id` |
| Profiler | `profile_data`, `profile_markdown` |
| Analyst | `insights`, `insights_markdown` |
| Evaluation | `critiques`, `critic_output`, `confidence_scores`, `uncertainty_output` |
| Visualizer | `visualization_result` |
| Reporter | `report_markdown` |
| Memory | `rag_analysis_context`, `rag_stored`, `rag_summary`, `memory_trace` |
| Observability | `langfuse_trace_id`, `graph_trace` |

## Deterministic vs LLM Responsibilities

The system intentionally separates deterministic work from generative work.

### Deterministic components

- `DataLoader` reads CSV, Excel, and Parquet files
- `DataProfiler` computes dataset statistics
- visualization parser validates LLM chart specs
- visualization executor runs chart code against pandas
- validators in `evaluation/validators.py` supply ground truth to the judge

### LLM-backed components

- Profiler agent writes the narrative profile
- Analyst generates insights
- Critic challenges insights
- Uncertainty Estimator scores semantic evidence
- Visualizer proposes charts
- Reporter writes the final executive report
- LLM Judge evaluates output quality after the run

## Model Routing

Model selection is centralized in `utils/llm.py`.

Default Ollama routing:

| Tier | Model | Typical use |
|---|---|---|
| `light` | `qwen3:4b` | Fast low-complexity tasks |
| `text` | `qwen3:14b` | Main reasoning and synthesis |
| `code` | `qwen2.5-coder:14b` | Chart-code generation |

Default task mapping:

- `profiler` -> `light`
- `analyst` -> `text`
- `critic` -> `text`
- `reporter` -> `text`
- `uncertainty` -> `light`
- `categorizer` -> `light`
- `visualizer` -> `code`

## Product Rendering Model

The Streamlit app uses two run modes:

- `stream_analysis()` for progressive UI updates
- final rerender after completion for the full React Flow pipeline diagram and tabs

Important UI behavior:

- live pipeline status uses plain HTML for stability during streaming
- final diagnostics pipeline diagram uses `streamlit_flow`
- charts are persisted as Plotly `figure_json` so they can be restored efficiently

## Storage and Observability

### Memory

The memory subsystem is built on ChromaDB:

- profiles, insights, reports, and Q&A can be stored
- retrieval is scoped by `dataset_id`
- the reporter retrieves prior context on demand
- `rag_storage` persists the current run after reporting

### Observability

There are two observability layers:

- internal diagnostics via `graph_trace`
- optional external GenAI tracing via LangFuse

LangFuse is self-hosted locally with Docker Compose and is fully optional.

## Architectural Strengths

- clear separation between orchestration, agents, and utilities
- centralized configuration and prompt loading
- explicit state contract
- progressive UI without duplicating business logic in the app layer
- optional tracing and memory that degrade gracefully when unavailable

## Current Boundaries

- the automated analysis pipeline is sequential, not parallel
- the Streamlit app is the primary deployment interface
- the system is optimized first for local Ollama use, with Gemini as an optional provider
