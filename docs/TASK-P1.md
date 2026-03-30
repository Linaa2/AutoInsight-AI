# P1 — Profiler + UI + Test: Technical Implementation

## Overview

P1 implements the first functional layer of AutoInsight-AI: deterministic dataset profiling, an LLM-powered profiler agent, and a Streamlit UI. It introduces no orchestration (that is P4) — the profiler and agent are called directly from the UI.

---

## File Structure

```
tools/
  data_loader.py        # File ingestion (CSV, Excel, Parquet)
  profiler_engine.py    # Deterministic pandas profiling
agents/
  profiler.py           # LLM agent: profile → markdown report
utils/
  llm.py                # LLM client factory (Ollama / Gemini)
config/
  prompts.yaml          # System and human prompts for the profiler agent
app/
  main.py               # Streamlit UI
tests/
  conftest.py           # Shared fixtures
  test_data_loader.py   # 9 tests for DataLoader
  test_profiler_engine.py # 25 tests for DataProfiler
data/
  sample_sales.csv      # 60-row e-commerce dataset for manual testing
```

---

## 1. Data Loading — `tools/data_loader.py`

### Class: `DataLoader`

Responsible for loading a pandas `DataFrame` from a file, regardless of whether it comes from disk or a browser upload.

#### Key design decisions

- **Auto-delimiter detection for CSV**: uses `pd.read_csv(sep=None, engine="python")` — no manual separator configuration needed.
- **Excel sheet selection via env**: the sheet to load is controlled by `DATA_LOADER_EXCEL_SHEET` (integer index or sheet name). Defaults to `0` (first sheet).
- **Byte-level ingestion**: `load_from_upload(file_bytes, file_name)` wraps bytes in `io.BytesIO` before passing to pandas — required for Streamlit's `UploadedFile` object.
- **Format detection**: resolved from the file extension via a static lookup dict (`SUPPORTED_EXTENSIONS`). Raises `UnsupportedFormatError` for unknown extensions.

#### Public interface

```python
loader = DataLoader()
df = loader.load("path/to/file.csv")               # from disk
df = loader.load_from_upload(raw_bytes, "file.xlsx")  # from upload
```

#### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `DATA_LOADER_EXCEL_SHEET` | `0` | Sheet index or name for Excel files |

---

## 2. Deterministic Profiling — `tools/profiler_engine.py`

### Classes: `DataProfiler`, `DataProfile`, `ColumnProfile`

Computes a fully deterministic statistical profile of any `DataFrame` using only pandas and numpy — no LLM involved.

### `ColumnProfile` — per-column statistics

Each column gets a `ColumnProfile` dataclass instance with fields adapted to its detected type:

| `dtype_category` | Detection logic | Fields populated |
|---|---|---|
| `boolean` | `pd.api.types.is_bool_dtype` | `top_values` (True/False counts) |
| `numeric` | `pd.api.types.is_numeric_dtype` | `min, max, mean, median, std, q25, q75, skewness, kurtosis, zeros_count, zeros_pct` |
| `datetime` | `pd.api.types.is_datetime64_any_dtype` | `min_date, max_date, date_range_days` |
| `categorical` | fallback (object/string) | `top_values` (top-N by frequency) |

> Boolean is checked before numeric because pandas encodes `bool` as a numeric dtype — without the early check, boolean columns would be misclassified.

### `DataProfile` — dataset-level statistics

Aggregates all column profiles plus global metrics:

```python
@dataclass
class DataProfile:
    shape: List[int]           # [rows, cols]
    duplicates_count: int
    duplicates_pct: float
    total_missing_count: int
    total_missing_pct: float
    memory_mb: float           # deep memory usage in MB
    numeric_cols: int
    categorical_cols: int
    datetime_cols: int
    boolean_cols: int
    other_cols: int
    columns: Dict[str, ColumnProfile]
    missing: Dict[str, float]  # col -> missing %
    samples: List[Dict]        # first N rows as records
    created_at: str            # ISO-8601 UTC timestamp
```

### `DataProfile.to_dict()` — output schema

Produces the canonical JSON structure expected by downstream consumers (the profiler agent and P2+ agents):

```json
{
  "shape": [1000, 12],
  "columns": {
    "revenue": {
      "name": "revenue",
      "dtype": "float64",
      "dtype_category": "numeric",
      "missing_count": 0,
      "missing_pct": 0.0,
      "unique_count": 46,
      "min": 39.99,
      "max": 1798.0,
      "mean": 395.001,
      "median": 225.375,
      "std": 412.26,
      "q25": 116.75,
      "q75": 498.25,
      "skewness": 1.575,
      "kurtosis": 1.722,
      "zeros_count": 0,
      "zeros_pct": 0.0
    }
  },
  "missing": {
    "customer_id": 3.33
  },
  "stats": {
    "duplicates_count": 0,
    "duplicates_pct": 0.0,
    "total_missing_count": 2,
    "total_missing_pct": 0.28,
    "memory_mb": 0.025,
    "numeric_cols": 4,
    "categorical_cols": 7,
    "datetime_cols": 0,
    "boolean_cols": 1,
    "other_cols": 0,
    "created_at": "2026-03-29T10:00:00+00:00"
  },
  "samples": [
    {"order_id": 1001, "customer_name": "Alice Martin", ...}
  ]
}
```

> The global metrics are grouped under `"stats"` rather than scattered at the top level, matching the schema defined in the task spec.

### `DataProfiler` — configuration via environment

```python
profiler = DataProfiler()
profile = profiler.profile(df)
```

| Variable | Default | Effect |
|---|---|---|
| `PROFILER_SAMPLE_ROWS` | `5` | Number of sample rows included in the profile |
| `PROFILER_TOP_VALUES` | `10` | Max number of top values shown for categorical/boolean columns |

### NaN / Inf safety

`_safe_float()` converts any numeric value to `float`, returning `None` for `NaN`, `Inf`, or unconvertible values. This ensures the profile is always JSON-serialisable without a custom encoder.

---

## 3. LLM Client — `utils/llm.py`

### Class: `LLMClient`

A factory that returns a LangChain-compatible `BaseChatModel` instance. The provider is selected at runtime via the `LLM_PROVIDER` environment variable — no code change is needed to switch between local and cloud inference.

```python
llm = LLMClient().get_llm()  # returns ChatOllama or ChatGoogleGenerativeAI
```

#### Provider selection

| `LLM_PROVIDER` | Class returned | Required env |
|---|---|---|
| `ollama` (default) | `ChatOllama` from `langchain_community` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |
| `gemini` | `ChatGoogleGenerativeAI` from `langchain_google_genai` | `GOOGLE_API_KEY`, `GEMINI_MODEL` |

Imports are deferred inside `_build_ollama` / `_build_gemini` — the unused provider's package is never imported, avoiding failures when only one is installed.

| Variable | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Which backend to use |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `LLM_TIMEOUT` | `60` | Request timeout in seconds |

---

## 4. Profiler Agent — `agents/profiler.py`

### Class: `ProfilerAgent`

Takes a `DataProfile` and returns a structured markdown report using an LLM. Built with a LangChain LCEL (LangChain Expression Language) chain.

```python
agent = ProfilerAgent()
markdown: str = agent.describe(profile)
```

#### Chain

```
ChatPromptTemplate  →  BaseChatModel  →  StrOutputParser
```

The prompt is populated with the full profile serialised as JSON (`profile.to_dict()`), giving the LLM access to every statistic computed by `DataProfiler`.

#### Prompt structure — `config/prompts.yaml`

The prompts are stored externally in YAML (not hardcoded) so they can be tuned without touching Python code. The path can be overridden via `PROMPTS_PATH`.

**System prompt** establishes the role and enforces output constraints:
- Professional, concise English
- Required section headings with emojis (enforced by listing them explicitly)
- Focus on meaning/implications, not raw number restatement
- Reference actual column names from the profile

**Human prompt** provides the profile JSON and per-section instructions:

| Section | Instruction |
|---|---|
| `## 📊 Dataset Overview` | Row/col count, memory, column type mix, one-sentence characterisation |
| `## 🔍 Data Quality Assessment` | Missing rate, columns >5% missing, duplicates, quality rating (Good/Fair/Poor) |
| `## 📋 Column-by-Column Analysis` | Grouped by type; stats, top values, and notable issues per column |
| `## 📈 Statistical Highlights` | 3–5 most interesting cross-column findings |
| `## 💡 Key Takeaways & Recommendations` | Exactly 3–5 actionable recommendations |

The prompt explicitly instructs the agent to handle any dataset type by adapting its analysis to what it finds in `columns` — it does not assume a specific domain.

---

## 5. Streamlit UI — `app/main.py`

### Layout

```
┌─ Sidebar ──────────────────┐  ┌─ Main area ───────────────────────────────────┐
│  📁 Upload Dataset          │  │  🔍 AutoInsight AI                            │
│  [file uploader]            │  │  [Rows] [Columns] [Duplicates] [Missing] [MB] │
│  ─────────────────          │  │  ───────────────────────────────────────────  │
│  ⚙️  Settings               │  │  📊 Overview │ 📋 Columns │ 🗂️ Sample │ 🤖 AI │
│  [AI Analysis toggle]       │  │                                               │
└─────────────────────────────┘  └───────────────────────────────────────────────┘
```

### Rendering functions

| Function | Tab | Content |
|---|---|---|
| `_render_kpi_row` | always visible | 5 `st.metric` cards: rows, cols, duplicates, missing %, memory |
| `_render_overview` | Overview | Column type breakdown + sortable missing-values table |
| `_render_columns` | Columns | One `st.expander` per column; adapts content to `dtype_category` |
| `_render_ai_analysis` | AI Analysis | Button → spinner → `st.markdown(description)` |

### Session state management

`st.session_state["ai_description"]` caches the LLM output so it is not regenerated on every Streamlit rerun. It is reset automatically when a new file is uploaded (detected by comparing `last_file` in session state).

### Error handling

- `UnsupportedFormatError` → `st.error` with the unsupported extension
- LLM failure → `st.error` + `st.info` with actionable fix hint (check Ollama / set API key)
- Generic load failure → `st.error` with the exception message

### sys.path injection

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

Inserted at the top of `app/main.py` so that `streamlit run app/main.py` resolves local imports (`tools`, `agents`, `utils`) without requiring the package to be installed.

---

## 6. Tests

### `tests/conftest.py` — shared fixtures

| Fixture | Type | Description |
|---|---|---|
| `sample_df` | `pd.DataFrame` | 20-row DataFrame with numeric, categorical, boolean columns and intentional missing values |
| `sample_csv_file` | `str` | Temp CSV path written from `sample_df` |
| `sample_excel_file` | `str` | Temp XLSX path written from `sample_df` |
| `sample_parquet_file` | `str` | Temp Parquet path written from `sample_df` |

### `tests/test_data_loader.py` — 9 tests

Covers: CSV / Excel / Parquet load from disk, upload bytes, `FileNotFoundError`, `UnsupportedFormatError`, integer sheet index via env, named sheet via env.

### `tests/test_profiler_engine.py` — 25 tests

| Category | Tests |
|---|---|
| Return types | `DataProfile` instance returned |
| Shape & global stats | shape, duplicate count, duplicate % |
| Missing values | per-column count, total %, zero-missing case |
| dtype_category | numeric, categorical, boolean, datetime |
| Numeric stats | all fields present, skewness/kurtosis, zeros count |
| Categorical stats | top_values present, respects `PROFILER_TOP_VALUES` env |
| Datetime stats | min/max date, date range in days |
| Samples | default count, custom count via env, string keys |
| Serialisation | `to_dict()` is JSON-serialisable, `shape` key correct |
| Edge cases | empty DataFrame, all-missing column |

### Running the tests

```bash
uv run pytest tests/ -v
# 34 passed in ~1s
```

---

## 7. Dependencies added

| Package | Version | Reason |
|---|---|---|
| `pyarrow` | `>=14.0.0` | Required by `pd.read_parquet()` |

`pyproject.toml` also gained:
- `[tool.setuptools.packages.find]` — fixes flat-layout discovery error during `pip install -e .`
- `[tool.pytest.ini_options]` — sets `pythonpath = ["."]` and `testpaths = ["tests"]` so pytest resolves local imports correctly

---

## 8. Environment variables summary

| Variable | Module | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | `utils/llm.py` | `ollama` | LLM backend: `ollama` or `gemini` |
| `OLLAMA_MODEL` | `utils/llm.py` | `mistral` | Ollama model name |
| `OLLAMA_BASE_URL` | `utils/llm.py` | `http://localhost:11434` | Ollama server URL |
| `GEMINI_MODEL` | `utils/llm.py` | `gemini-1.5-flash` | Gemini model name |
| `GOOGLE_API_KEY` | `utils/llm.py` | — | Required when `LLM_PROVIDER=gemini` |
| `LLM_TIMEOUT` | `utils/llm.py` | `60` | LLM request timeout (seconds) |
| `PROFILER_SAMPLE_ROWS` | `tools/profiler_engine.py` | `5` | Sample rows in profile |
| `PROFILER_TOP_VALUES` | `tools/profiler_engine.py` | `10` | Top values for categorical columns |
| `DATA_LOADER_EXCEL_SHEET` | `tools/data_loader.py` | `0` | Excel sheet index or name |
| `PROMPTS_PATH` | `agents/profiler.py` | `config/prompts.yaml` | Override path to prompts file |
| `APP_TITLE` | `app/main.py` | `AutoInsight AI` | Streamlit page title |
