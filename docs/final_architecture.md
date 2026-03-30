# AutoInsight-AI — Final Architecture

## Overview

AutoInsight-AI is a multi-agent data analysis product built with LangGraph.
Users upload a dataset and receive a progressive, coordinated analysis:
deterministic profiling → LLM-powered insights → auto-generated charts →
executive report.

## Directory Roles

| Package           | Role                                       |
|-------------------|--------------------------------------------|
| `orchestration/`  | **Canonical** graph state + graph builder   |
| `agents/`         | LLM agent classes (no deterministic logic)  |
| `tools/`          | Deterministic utilities (loader, profiler)  |
| `visualization/`  | Chart schemas, parser, sandboxed executor   |
| `config/`         | Settings, prompt templates (single source)  |
| `utils/`          | LLM client factory, prompt loader, memory   |
| `app/`            | Streamlit UI (no business logic)            |
| `evaluation/`     | Scoring / validation (future)               |
| `graph/`          | Thin compatibility wrapper → `orchestration`|

## Canonical Orchestration

The single authoritative orchestration layer lives in `orchestration/`.

- **`orchestration/state.py`** — `PipelineState` TypedDict + `NodeTraceEntry`
- **`orchestration/graph.py`** — Node functions, `build_graph()`, `run_analysis()`

The `graph/` package re-exports from `orchestration/` for backward compatibility.

### State Contract

```python
class PipelineState(TypedDict, total=False):
    # inputs
    df_dict: list[dict[str, Any]]
    file_name: str
    # profiler
    profile_data: dict[str, Any]
    profile_markdown: str
    # analyst
    insights: list[dict[str, Any]]
    insights_markdown: str
    # visualizer
    visualization_result: dict[str, Any]
    # reporter
    report_markdown: str
    # diagnostics
    graph_trace: list[NodeTraceEntry]
    error: str
```

### Graph Flow

```
START → profiler_node → analyst_node → visualizer_node → reporter_node → END
```

Each node:
- Reads only the keys it needs
- Writes only the keys it produces
- Appends a `NodeTraceEntry` to `graph_trace`
- Fails gracefully (writes error, never raises)

## Agent Coordination

1. **profiler_node** — runs `DataProfiler` (deterministic) then `ProfilerAgent` (LLM)
2. **analyst_node** — takes profile output + data sample → `AnalystAgent` → structured insights
3. **visualizer_node** — takes profile + insights → `VisualizerAgent` → chart specs → executor
4. **reporter_node** — takes all prior outputs → `ReporterAgent` → executive report

## Adding a Future Node

1. Create `agents/my_agent.py` with a class that accepts pipeline data and returns results.
2. Add a node function in `orchestration/graph.py` that reads/writes `PipelineState`.
3. Add any new state keys to `PipelineState` in `orchestration/state.py`.
4. Wire the node into the graph in `build_graph()`.

## Configuration

- **Static paths**: `config.settings.REPO_ROOT`, `PROMPTS_PATH`, etc.
- **Runtime settings**: `config.settings.settings` (from `.env`)
- **Prompts**: `config/prompts.yaml` → loaded via `utils.prompt_loader.load_prompt_section()`
- **LLM**: `utils.llm.LLMClient` — task-aware (text vs code model)

No agent or tool computes its own prompt path or model name.

## Graph Monitoring

Every node records a `NodeTraceEntry` with:
- node name, status (success/failed/skipped)
- start/end timestamps, duration
- keys read and written
- summary and error message

The trace is accumulated in `graph_trace` and displayed in the
Diagnostics tab of the Streamlit app.
