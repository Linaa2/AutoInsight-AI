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
    df_dict: list[dict[str, Any]]        # JSON-safe dataset
    file_name: str

    # Profiler
    profile_data: dict[str, Any]         # raw DataProfile.to_dict()
    profile_markdown: str                # LLM-generated markdown report

    # Analyst (Phase 2)
    insights_markdown: str

    # Visualizer
    visualization_result: dict[str, Any] # serialised chart specs + metadata

    # Reporter (future)
    report: str

    # Critic (future)
    feedback: str

    # Diagnostics
    error: str
```

**Why `df_dict` instead of `DataFrame`?** LangGraph serialises state for
checkpointing. A list-of-dicts is JSON-safe. Nodes reconstruct the DataFrame
locally via `pd.DataFrame(state["df_dict"])`.

---

## Current graph

```
START → profiler_node → visualizer_node → END
```

### profiler_node

1. Reconstructs the DataFrame from `df_dict`.
2. Runs `DataProfiler().profile(df)` (deterministic stats).
3. Runs `ProfilerAgent().describe(profile)` (LLM markdown).
4. Writes `profile_data` and `profile_markdown`.

### visualizer_node

1. Reads `profile_markdown` and `insights_markdown`.
2. If `insights_markdown` is absent, falls back to `profile_markdown`
   (useful for testing before the Analyst agent exists).
3. Runs `run_visualization_pipeline(df, request)`.
4. Serialises the result (without Plotly figure objects) into
   `visualization_result`.

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
