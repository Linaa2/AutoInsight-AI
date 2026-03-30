# AutoInsight-AI — Architecture & Tasks (P1–P4)

> A multi-agent data analysis product built with **LangGraph** and **Streamlit**.
> Users upload a dataset and receive a progressive, coordinated analysis:
> deterministic profiling → LLM-powered insights → auto-generated charts → executive report.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Phase 1 — Profiler Agent & UI](#phase-1--profiler-agent--ui)
3. [Phase 2 — Analyst Agent](#phase-2--analyst-agent)
4. [Phase 3 — Visualizer Agent](#phase-3--visualizer-agent)
5. [Phase 4 — Reporter Agent & LangGraph Orchestration](#phase-4--reporter-agent--langgraph-orchestration)
6. [Unified Pipeline & State Contract](#unified-pipeline--state-contract)
7. [Streamlit Product](#streamlit-product)
8. [Configuration & Infrastructure](#configuration--infrastructure)
9. [Testing Strategy](#testing-strategy)
10. [Project Structure](#project-structure)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app/)                      │
│  Upload → KPI Preview → Progressive Tabs → Download Results │
└─────────────────────────┬───────────────────────────────────┘
                          │ stream_analysis(df, file_name)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Orchestration (orchestration/)        │
│                                                             │
│  START → profiler_node → analyst_node → visualizer_node     │
│                                    → reporter_node → END    │
│                                                             │
│  PipelineState (TypedDict) flows between nodes              │
│  NodeTraceEntry recorded per node for diagnostics           │
└──────┬──────────┬──────────┬──────────┬─────────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
   Profiler    Analyst   Visualizer  Reporter
    Agent       Agent      Agent      Agent
   (agents/)   (agents/)  (agents/)  (agents/)
       │          │          │
       ▼          │          ▼
  DataProfiler    │    Parser + Executor
  DataLoader      │    (visualization/)
   (tools/)       │
                  │
           ┌──────┴──────┐
           │  LLM Client │  (utils/llm.py)
           │  Ollama or   │
           │  Gemini      │
           └──────────────┘
```

**Design principles:**
- **Deterministic first** — all statistics computed via pandas; LLMs explain and interpret.
- **Modular agents** — each agent has a single responsibility with typed I/O.
- **Flat state machine** — LangGraph `PipelineState` with no nested objects.
- **Safe execution** — chart code runs in a restricted sandbox.
- **Graceful failure** — every node catches exceptions, writes errors, never raises.

---

## Phase 1 — Profiler Agent & UI

**Goal:** Deterministic dataset profiling with LLM-powered interpretation.

### Components

| Module | Role |
|--------|------|
| `tools/data_loader.py` | Ingests CSV, Excel, Parquet with auto-delimiter detection |
| `tools/profiler_engine.py` | Deterministic pandas profiling → `DataProfile` dataclass |
| `agents/profiler.py` | LLM interprets raw profile → markdown report |
| `app/main.py` | Streamlit UI: upload, KPI row, profile tab |

### Data Loader

Supports `.csv` (auto-detect separator), `.xlsx` / `.xls` (sheet selection via `EXCEL_SHEET`), and `.parquet`. Accepts both file paths and raw bytes (for Streamlit uploads). Validates format and raises `UnsupportedFormatError` for anything else.

### Profiler Engine

`DataProfiler.profile(df)` produces a `DataProfile` with:

- **Dataset-level:** shape, duplicate count, total missing %, memory usage
- **Per-column:** dtype-aware statistics
  - Numeric: min, max, mean, median, std, skewness, kurtosis, missing count
  - Categorical: unique count, top values (configurable via `PROFILER_TOP_N`)
  - Datetime: min/max date range
  - Boolean: true/false/null counts
- **Sample rows** for LLM context (configurable via `PROFILER_SAMPLE_ROWS`)

Output is JSON-serializable via `DataProfile.to_dict()`.

### Profiler Agent

Takes the structured `DataProfile` and calls the text LLM with a prompt from `config/prompts.yaml` to generate a markdown report with sections:

1. 📊 Dataset Overview
2. 🔍 Data Quality Assessment
3. 📋 Column-by-Column Analysis
4. 📈 Statistical Highlights
5. 💡 Key Takeaways & Recommendations

### Tests (P1)

- **DataLoader:** 9 tests covering all formats, upload bytes, sheet selection, unsupported format errors
- **DataProfiler:** 16 tests — shape, columns, duplicates, missing values, numeric/categorical/datetime stats, JSON serialization, edge cases (empty df, all-missing column)

---

## Phase 2 — Analyst Agent

**Goal:** Generate structured, categorized business insights from the dataset profile.

### Components

| Module | Role |
|--------|------|
| `agents/analyst.py` | LLM generates insights → parsed + validated + categorized |
| `agents/mock_profiler.py` | Mock state for testing without running the profiler |

### Insight Structure

Each insight is a dictionary with:

```json
{
  "title": "High concentration in Ile-de-France",
  "observation": "32% of orders come from one region",
  "hypothesis": "Population density and logistics proximity",
  "recommendation": "Expand into underserved regions",
  "priority": "high",
  "category": "distribution"
}
```

### Pipeline Stages

1. **LLM call** — sends profiler markdown + data sample to the text LLM with the analyst prompt
2. **JSON extraction** (`extract_json`) — parses JSON from noisy LLM output (handles markdown fences, prose)
3. **Fallback** (`fallback_parse_markdown`) — if JSON extraction fails, regex-parses structured markdown
4. **Validation** (`validate_insight`) — ensures all required fields are non-empty
5. **Priority normalization** — maps French/English priority terms to `high` / `medium` / `low`
6. **Categorization** (`InsightCategorizer`) — keyword-based (default, fast) or LLM-based (optional)
7. **Formatting** (`InsightFormatter.to_markdown`) — groups insights by category with priority icons

### Insight Categories

| Category | Icon | Detects |
|----------|------|---------|
| Trend | 📈 | Temporal evolution, growth, decline |
| Anomaly | ⚠️ | Outliers, unexpected values, spikes |
| Correlation | 🔗 | Relationships between variables |
| Distribution | 📊 | Concentration, spread, segmentation |
| General | 💡 | Other findings |

### Tests (P2)

- **Unit:** JSON extraction, field validation, priority normalization, keyword categorization, batch categorization, markdown fallback, formatting, empty list handling
- **Integration** (marked `@pytest.mark.integration`): Full pipeline with Ollama LLM, LangGraph node entry point

---

## Phase 3 — Visualizer Agent

**Goal:** Auto-generate executable Plotly charts based on dataset structure and insights.

### Components

| Module | Role |
|--------|------|
| `agents/visualizer.py` | Orchestrates LLM call → parsing → execution |
| `visualization/schemas.py` | Dataclass contracts: `ChartSpec`, `VisualizerRequest`, `VisualizerResult` |
| `visualization/parser.py` | Extracts & validates chart JSON from LLM response |
| `visualization/executor.py` | Runs chart code in a restricted Python sandbox |

### Chart Specification

The code LLM generates a JSON array of chart specs:

```json
{
  "title": "Sales by Region",
  "chart_type": "bar",
  "code": "fig = px.bar(df, x='region', y='sales', title='Sales by Region')",
  "explanation": "Shows regional sales distribution",
  "columns_used": ["region", "sales"]
}
```

Allowed chart types: `bar`, `line`, `scatter`, `histogram`, `box`, `pie`, `heatmap`, `area`, `violin`, `sunburst`, `treemap`, `funnel`, `waterfall`, `bubble`.

### Sandboxed Execution

`execute_chart(df, code)` runs the LLM-generated code in an isolated namespace:

- **Allowed globals:** `pd` (pandas), `np` (numpy), `px` (plotly.express), `go` (plotly.graph_objects), `df` (the dataset)
- **Forbidden:** `import`, `open`, `exec`, `eval`, `os`, `sys`, `subprocess`, file I/O
- Returns an `ExecutionResult` with `success`, `figure`, and `error` fields

### Pipeline Flow

```
VisualizerRequest → LLM (code model) → raw JSON string
    → Parser (extract + validate specs) → list[ChartSpec]
    → Executor (sandbox each spec) → list[RenderedChart]
    → VisualizerResult (charts + metadata)
```

### Tests (P3)

- **Parser:** 14 tests — valid JSON, markdown fences, optional fields, all chart types, malformed JSON, truncated JSON, missing fields, invalid types, partial success, nested JSON in prose
- **Executor:** 11 tests — bar/scatter/histogram/line/box/graph_objects charts, syntax errors, missing columns, no figure assigned, empty code, forbidden imports, figure type validation

---

## Phase 4 — Reporter Agent & LangGraph Orchestration

**Goal:** Synthesize all prior outputs into an executive-ready markdown report, coordinated through a LangGraph state graph.

### Reporter Agent

| Module | Role |
|--------|------|
| `agents/reporter.py` | Collects P1–P3 outputs → LLM generates executive report |

Takes as input:
- Profiler markdown (P1)
- Analyst markdown + structured insights (P2)
- Visualizer chart metadata (P3)

Produces a markdown report with sections:

1. 📝 **Executive Summary** — 3–5 sentences for decision-makers
2. 📊 **Dataset Description** — source, size, quality, missing values
3. 🔍 **Key Insights** — narratively reformulated insights grouped by theme/priority
4. 📈 **Visualizations** — interpretation of each chart
5. ✅ **Recommendations** — 3–5 concrete, actionable items
6. ⚠️ **Limitations & Next Steps** — what the analysis doesn't cover

Helper functions:
- `extract_chart_titles(viz_output)` — pulls chart titles for the report
- `build_charts_summary(viz_output)` — formats chart list for the prompt
- `build_insights_summary(insights)` — formats insights with priority/category for the prompt

### LangGraph Orchestration

The canonical orchestration lives in `orchestration/`:

**State** (`orchestration/state.py`):

```python
class PipelineState(TypedDict, total=False):
    df_dict: list[dict[str, Any]]     # serialized DataFrame
    file_name: str
    profile_data: dict[str, Any]       # raw profile
    profile_markdown: str              # LLM-interpreted profile
    insights: list[dict[str, Any]]     # structured insights
    insights_markdown: str             # formatted insights
    visualization_result: dict[str, Any]  # chart specs + execution results
    report_markdown: str               # final report
    graph_trace: list[NodeTraceEntry]  # per-node diagnostics
    error: str
```

**Graph** (`orchestration/graph.py`):

```
START → profiler_node → analyst_node → visualizer_node → reporter_node → END
```

Each node function:
1. Reads only the state keys it needs
2. Delegates to the corresponding agent class
3. Writes its output keys + appends a `NodeTraceEntry`
4. Catches all exceptions and records them (never raises)

**Entry points:**
- `run_analysis(df, file_name)` — runs synchronously, returns final `PipelineState`
- `stream_analysis(df, file_name)` — yields `(node_name, node_output)` after each node completes

### Graph Tracing

Every node records a `NodeTraceEntry`:

```python
class NodeTraceEntry(TypedDict):
    node: str                # "profiler", "analyst", etc.
    status: str              # "success", "failed", "skipped"
    started_at: str          # ISO timestamp
    finished_at: str         # ISO timestamp
    duration_s: float        # wall-clock seconds
    keys_read: list[str]     # state keys consumed
    keys_written: list[str]  # state keys produced
    summary: str             # human-readable description
    error: str | None        # error message if failed
```

### Tests (P4)

- **Reporter unit:** chart title extraction (including empty), chart summary formatting, insights summary formatting
- **Reporter integration** (marked `@pytest.mark.integration`): full LLM pipeline, LangGraph node
- **Smoke:** import validation for all packages and orchestration module

---

## Unified Pipeline & State Contract

The 4 phases compose into a single linear pipeline. Data flows through `PipelineState`:

```
┌──────────┐    profile_data     ┌──────────┐     insights       ┌────────────┐  visualization_result  ┌──────────┐
│ Profiler  │──  profile_markdown──▶│ Analyst  │── insights_markdown──▶│ Visualizer │────────────────────────▶│ Reporter  │
│   Node    │                    │   Node   │                    │    Node    │                        │   Node   │
└──────────┘                    └──────────┘                    └────────────┘                        └──────────┘
     ▲                                                                                                      │
     │  df_dict, file_name                                                          report_markdown         │
     │                                                                                                      ▼
   START                                                                                                   END
```

Each node is **independently testable** and **fault-tolerant**:
- If the profiler fails, downstream nodes skip gracefully
- If the analyst fails, the visualizer falls back to profile-only input
- The reporter produces whatever it can from available outputs

---

## Streamlit Product

### User Experience Flow

1. **Landing** — welcome message with agent descriptions
2. **Upload** — sidebar file uploader (CSV / Excel / Parquet)
3. **Preview** — KPI row (rows, columns, duplicates, missing %, memory) + data expander
4. **Run** — "🚀 Run Full Analysis" button triggers `stream_analysis()`
5. **Progressive results** — tabs appear and fill in real-time as each agent completes
6. **Final results** — five tabs: Profile, Insights, Visualizations, Report, Diagnostics

### Progressive Rendering

During execution:
- A **Lottie animation** plays above the result area
- A **progress bar** and **status pills** show which agents have completed
- **Tabs render progressively** — completed agent tabs show full results while pending tabs show loading indicators
- A **Stop button** allows cancelling the analysis mid-run

After completion:
- All tabs are fully rendered with final content
- A **Re-run button** appears in the sidebar
- **Download buttons** are available on every tab (`.md`, `.json`, `.txt`)

### Key UI Decisions

- Chart figures are **re-executed at render time** (Plotly figures aren't JSON-serializable)
- Session state is **keyed per file name** — uploading a different file resets all results
- The UI layer contains **no business logic** — it delegates entirely to `orchestration.graph`

---

## Configuration & Infrastructure

### Settings (`config/settings.py`)

| Setting | Source | Purpose |
|---------|--------|---------|
| `LLM_PROVIDER` | `.env` | `"ollama"` (default) or `"gemini"` |
| `OLLAMA_TEXT_MODEL` | `.env` | Text/reasoning model (e.g. `mistral`) |
| `OLLAMA_CODE_MODEL` | `.env` | Code generation model (e.g. `codellama`) |
| `LLM_TIMEOUT` | `.env` | Request timeout in seconds |
| `PROFILER_TOP_N` | `.env` | Number of top values for categorical columns |
| `PROFILER_SAMPLE_ROWS` | `.env` | Number of sample rows for LLM context |

### LLM Client (`utils/llm.py`)

Task-aware factory:
- `LLMClient.get_text_llm()` → reasoning model (profiler, analyst, reporter)
- `LLMClient.get_code_llm()` → code generation model (visualizer)

Supports provider switching: Ollama (local, default) or Google Gemini (cloud fallback).

### Prompts (`config/prompts.yaml`)

Centralized prompt templates loaded via `utils.prompt_loader.load_prompt_section(key)`. Sections:
- `profiler` — dataset interpretation prompt
- `analyst` — insight generation prompt
- `visualizer` — chart code generation prompt
- `reporter` — report synthesis prompt
- `categorizer` — insight categorization prompt

---

## Testing Strategy

### Test Organization

| File | Tests | Type | Speed |
|------|-------|------|-------|
| `test_data_loader.py` | 9 | Unit | < 1s |
| `test_profiler_engine.py` | 16 | Unit | < 1s |
| `test_analyst.py` | 8 unit + 2 integration | Mixed | Unit < 1s, Integration 15–60s |
| `test_reporter.py` | 6 unit + 2 integration | Mixed | Unit < 1s, Integration 15–60s |
| `test_visualizer_parser.py` | 14 | Unit | < 1s |
| `test_visualizer_executor.py` | 11 | Unit | < 1s |
| `test_smoke.py` | 2 | Smoke | < 1s |

### Running Tests

```bash
make test          # Unit tests only (fast, ~1s) — excludes @pytest.mark.integration
make test-all      # All tests including LLM integration tests (requires Ollama)
make check         # Full check: lint + format + unit tests + pre-commit hooks
```

Integration tests are marked with `@pytest.mark.integration` and require a running Ollama instance. They are excluded from the default `make test` / `make check` targets to keep the feedback loop fast.

---

## Project Structure

```
AutoInsight-AI/
├── agents/                 # LLM agent classes
│   ├── profiler.py         #   P1: dataset interpretation
│   ├── analyst.py          #   P2: insight generation
│   ├── visualizer.py       #   P3: chart code generation
│   ├── reporter.py         #   P4: report synthesis
│   └── mock_profiler.py    #   mock state for testing
├── app/
│   └── main.py             # Streamlit UI (progressive rendering)
├── orchestration/          # Canonical LangGraph orchestration
│   ├── state.py            #   PipelineState + NodeTraceEntry
│   └── graph.py            #   Node functions + graph builder
├── tools/                  # Deterministic utilities
│   ├── data_loader.py      #   File ingestion (CSV/Excel/Parquet)
│   └── profiler_engine.py  #   Pandas-based profiling
├── visualization/          # Chart pipeline
│   ├── schemas.py          #   Dataclass contracts
│   ├── parser.py           #   JSON extraction from LLM output
│   └── executor.py         #   Sandboxed code execution
├── config/
│   ├── settings.py         #   Static paths + runtime config
│   └── prompts.yaml        #   Centralized prompt templates
├── utils/
│   ├── llm.py              #   LLM client factory
│   ├── prompt_loader.py    #   YAML prompt section loader
│   └── memory.py           #   Conversation memory (ChromaDB)
├── tests/                  #   77 unit + 4 integration tests
├── assets/                 #   Logo, Lottie animation, images
├── docs/                   #   Documentation
├── graph/                  #   Compat wrapper → orchestration/
├── evaluation/             #   Scoring / validation (future)
├── data/                   #   Sample datasets
├── pyproject.toml          #   Dependencies + tool config
└── Makefile                #   Dev commands
```
