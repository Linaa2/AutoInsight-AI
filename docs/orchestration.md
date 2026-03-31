# LangGraph Orchestration

AutoInsight-AI uses [LangGraph](https://github.com/langchain-ai/langgraph) to
orchestrate its multi-agent pipeline. The architecture is modular: each agent
is a graph **node** that reads from and writes to a shared state object.

---

## Structure

```
orchestration/
├── __init__.py
├── state.py      # PipelineState TypedDict — shared graph contract
└── graph.py      # Node functions + build_graph()
```

---

## PipelineState

Defined in `orchestration/state.py` as a `TypedDict(total=False)`.

All fields are optional so that each node only returns the keys it produces.

```python
class PipelineState(TypedDict, total=False):
    df_ref: str                          # in-process DataFrame handle (preferred)
    df_dict: list[dict[str, Any]]        # legacy JSON-safe fallback
    file_name: str
    dataset_id: str

    # Profiler
    profile_data: dict[str, Any]         # raw DataProfile.to_dict()
    profile_markdown: str                # LLM-generated markdown report

    # Analyst (Phase 2)
    insights_markdown: str

    # Visualizer
    visualization_result: dict[str, Any] # serialised chart specs + metadata

    # Reporter
    report_markdown: str

    # RAG / memory
    rag_analysis_context: str
    rag_stored: bool
    rag_summary: str
    memory_trace: list[dict[str, Any]]

    # Diagnostics
    graph_trace: list[dict[str, Any]]
    error: str
```

**Why `df_ref`?** For the normal app path, the DataFrame is registered once in
process and nodes access it by lightweight reference. This avoids repeated
`to_dict()` / `DataFrame(...)` conversions on large datasets.

**Why keep `df_dict`?** Backward compatibility: external callers can still
invoke the graph with a JSON-safe dataset payload when needed.

---

## Current graph

```
START → profiler_node → analyst_node → visualizer_node → reporter_node → rag_storage_node → END
```

### profiler_node

1. Reconstructs the DataFrame from `df_dict`.
2. Runs `DataProfiler().profile(df)` (deterministic stats).
3. Runs `ProfilerAgent().describe(profile)` (LLM markdown).
4. Writes `profile_data` and `profile_markdown`.

### analyst_node

1. Reads `profile_markdown`, `profile_data`, and a small data sample.
2. Runs `AnalystAgent().run(...)`.
3. Writes `insights` and `insights_markdown`.

### visualizer_node

1. Reads `profile_markdown` and `insights_markdown`.
2. Runs `run_visualization_pipeline(df, request)`.
3. Serialises chart specs plus execution metadata into `visualization_result`.

### reporter_node

1. Reads profiler / analyst / visualizer outputs.
2. Retrieves prior RAG context on demand (best-effort) right before report generation.
3. Runs `ReporterAgent().run(...)`.
4. Writes `report_markdown` and, when available, `rag_analysis_context`.

### rag_storage_node

1. Stores profile / insights / report outputs in ChromaDB (best-effort).
2. Records one `MemoryTraceEntry` per storage action.

---

## Usage

```python
import pandas as pd
from orchestration.graph import build_graph

df = pd.read_csv("data/sample_sales.csv")
graph = build_graph()

result = graph.invoke({
    "df_dict": df.to_dict(orient="records"),
    "file_name": "sample_sales.csv",
})

print(result["profile_markdown"])
print(result["visualization_result"])
```

---

## Adding a new agent

To add an Analyst agent (Phase 2):

1. **Create the agent** in `agents/analyst.py` following the same pattern as
   `ProfilerAgent` — load prompts via `load_prompt_section("analyst")`, use
   `LLMClient.get_text_llm()`.

2. **Add prompts** to `config/prompts.yaml` under the `analyst:` key.

3. **Add a node function** in `orchestration/graph.py`:

   ```python
   def analyst_node(state: PipelineState) -> PipelineState:
       df = pd.DataFrame(state["df_dict"])
       profile_md = state.get("profile_markdown", "")
       # ... run analyst ...
       return {"insights_markdown": insights}
   ```

4. **Wire it into the graph** in `build_graph()`:

   ```python
   graph.add_node("analyst", analyst_node)

   graph.add_edge(START, "profiler")
   graph.add_edge("profiler", "analyst")
   graph.add_edge("analyst", "visualizer")
   graph.add_edge("visualizer", END)
   ```

5. **Add fields** to `PipelineState` if needed (it already has
   `insights_markdown` for the analyst).

The same pattern applies for Reporter, Critic, or any future agent.

---

## Design principles

- **Agents stay decoupled** — they know nothing about the graph. They receive
  typed inputs and return typed outputs. The graph wires them together.
- **State is flat** — no nested objects. Each node reads/writes top-level keys.
- **Nodes never raise** — errors are captured in the `error` field.
- **Extensible** — adding a node is a 4-step recipe (agent, prompts, node
  function, edge).
