# Phase 3: Visualizer Agent

The Visualizer Agent (`agents/visualizer.py`) is Phase 3 of the AutoInsight-AI pipeline.
It takes structured context from the Profiler and Analyst agents and produces rendered
Plotly charts by asking a code-generation LLM to write the visualization code.

---

## Architecture

```
VisualizerRequest
      │
      ▼
VisualizerAgent._chain   ← ChatPromptTemplate (system + human from prompts.yaml)
      │                     │
      │                     └─ LLMClient.get_code_llm()  ← OLLAMA_CODE_MODEL
      ▼
  raw LLM text (JSON)
      │
      ▼
parse_llm_output()       ← visualization/parser.py
      │
      ▼
list[ChartSpec]
      │
      ▼
execute_chart()          ← visualization/executor.py  (restricted exec sandbox)
      │
      ▼
VisualizationPipelineResult
```

---

## Data Contracts

All contracts are stdlib `dataclasses` defined in `visualization/schemas.py`.

| Class | Role |
|---|---|
| `VisualizerRequest` | Input: profile + insights + column list |
| `ChartSpec` | One chart: title, chart_type, code, explanation, columns_used |
| `VisualizerLLMOutput` | Parsed list of `ChartSpec` objects |
| `ChartExecutionResult` | Success flag, Plotly figure, error string |
| `RenderedChart` | `ChartSpec` paired with its `ChartExecutionResult` |
| `VisualizationPipelineResult` | Full output: list of `RenderedChart` + metadata |

---

## LLM Model Routing

`LLMClient` (`utils/llm.py`) routes to different models based on task type:

| Method | Env var | Default | Use case |
|---|---|---|---|
| `get_text_llm()` | `OLLAMA_TEXT_MODEL` | `qwen3:14b` | Natural language reasoning |
| `get_code_llm()` | `OLLAMA_CODE_MODEL` | `qwen2.5-coder:14b` | Code generation |

The Visualizer Agent uses `get_code_llm()` because it generates executable Python.

Set `LLM_PROVIDER=gemini` to route to Google Gemini instead of Ollama.

---

## Prompt Structure

Templates live in `config/prompts.yaml` under the `visualizer` key.

```yaml
visualizer:
  system: |          # Static role + JSON format rules
  human: |           # Dynamic: {columns_info}, {profile_markdown}, {insights_markdown}
```

`ChatPromptTemplate.from_messages` loads both parts. Literal `{` / `}` in the
system template are escaped as `{{` / `}}` following ChatPromptTemplate convention.

---

## Execution Sandbox

`execute_chart(df, code)` runs LLM-generated code in a restricted namespace:

```python
namespace = {
    "__builtins__": {},   # no imports, no open(), no eval()
    "df": df,             # the caller's DataFrame
    "pd": pandas,
    "px": plotly.express,
    "go": plotly.graph_objects,
    "np": numpy,
}
exec(code, namespace)
fig = namespace.get("fig")  # must be assigned by the generated code
```

With `__builtins__ = {}`:
- `import` statements in generated code **raise** and are caught cleanly.
- Dangerous built-ins (`open`, `__import__`, `eval`) are unavailable.
- Simple one-liners (`fig = px.bar(df, ...)`) work without builtins.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `"ollama"` or `"gemini"` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TEXT_MODEL` | `qwen3:14b` | Model for text/reasoning tasks |
| `OLLAMA_CODE_MODEL` | `qwen2.5-coder:14b` | Model for code-generation tasks |
| `LLM_TIMEOUT` | `60` | Request timeout in seconds |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name (when provider=gemini) |
| `GOOGLE_API_KEY` | — | Required when `LLM_PROVIDER=gemini` |

Copy `.env.example` to `.env` and fill in values before running.

> **Apple Silicon note**: Ollama handles Metal/MPS acceleration internally.
> Do not pass `device="mps"` to any LangChain model — it is not a supported parameter.

---

## Public API

```python
from agents.visualizer import (
    VisualizerAgent,            # class — for direct use or dependency injection
    run_visualization_pipeline, # full pipeline: LLM → parse → exec
    generate_visualizations,    # backward-compatible wrapper
)
from visualization.schemas import VisualizerRequest, VisualizationPipelineResult
```

### High-level usage (recommended)

```python
import pandas as pd
from agents.visualizer import generate_visualizations

df = pd.read_csv("my_dataset.csv")
result = generate_visualizations(
    df=df,
    profile_summary="Dataset contains 5 000 rows...",
    insights_text="Sales peak in Q4. Region East outperforms others.",
    columns_info="region (object), sales (float64), quarter (int64)",
)

for chart in result.charts:
    if chart.execution.success:
        chart.execution.figure.show()
```

### With explicit request object

```python
from agents.visualizer import run_visualization_pipeline
from visualization.schemas import VisualizerRequest

request = VisualizerRequest(
    profile_markdown="...",
    insights_markdown="...",
    columns_info="region (object), sales (float64)",
)
result = run_visualization_pipeline(df, request)
```

### With dependency injection (testing)

```python
from unittest.mock import MagicMock
from agents.visualizer import VisualizerAgent
from visualization.schemas import VisualizerRequest

mock_llm = MagicMock()
mock_llm.invoke.return_value.content = '{"charts": []}'

agent = VisualizerAgent(llm=mock_llm)
raw = agent.generate_raw(VisualizerRequest(...))
```

---

## Error Handling

The pipeline never raises. All failures are captured in `VisualizationPipelineResult`:

| Failure mode | Where it appears |
|---|---|
| LLM unreachable / timeout | `parsing_error = "LLM call failed: ..."`, `charts = []` |
| LLM returns non-JSON | `parsing_error = "Could not find a JSON object..."` |
| Some chart specs invalid | `parsing_error` describes each failure; valid charts still returned |
| Chart code crashes at exec | `chart.execution.success = False`, `chart.execution.error` has details |
