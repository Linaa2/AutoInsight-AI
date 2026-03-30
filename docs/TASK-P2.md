# Phase 2: Analyst Agent

## Overview

The Analyst agent (`agents/analyst.py`) generates structured, categorized insights from a dataset profile. It takes the output of the Profiler (Phase 1) and produces actionable business insights.

## Architecture

```
agents/analyst.py         → Core agent + helpers + LangGraph node
agents/mock_profiler.py   → Mock state for testing without Phase 1
config/prompts.yaml       → LLM prompts (analyst, categorizer sections)
app/main.py               → Streamlit UI (Insights tab)
tests/test_analyst.py     → Unit + integration tests
```

### Classes

| Class | Role |
|---|---|
| `AnalystAgent` | Core agent — orchestrates generation, categorization, formatting |
| `InsightCategorizer` | Categorizes insights via keyword matching or LLM |
| `InsightFormatter` | Converts insights to grouped markdown with icons |

### Helper Functions

| Function | Role |
|---|---|
| `extract_json()` | Extracts JSON from noisy LLM responses (handles backticks, extra text) |
| `validate_insight()` | Checks all required fields are present and non-empty |
| `normalize_priority()` | Normalizes priority strings to `high` / `medium` / `low` |
| `fallback_parse_markdown()` | Parses insights from markdown when JSON parsing fails |
| `analyst_node()` | LangGraph node entry point (delegates to `AnalystAgent`) |

## Pipeline

```
profiler_output + sample_text + profile_data
        │
        ▼
  ┌─────────────┐
  │  AnalystAgent│
  │   .run()     │
  └──────┬───────┘
         │
    1. _generate()        → Sends structured prompt to LLM, receives raw response
    2. _parse_response()  → extract_json() or fallback_parse_markdown()
    3. validate_insight() → Filters out malformed insights
    4. normalize_priority()
    5. categorizer.categorize_all()  → Keyword-based or LLM categorization
    6. formatter.to_markdown()       → Grouped markdown output
         │
         ▼
  { "analyst_output": str, "insights": list[dict] }
```

## Input (from LangGraph state)

| Key | Type | Description |
|---|---|---|
| `profiler_output` | `str` | Markdown report from the Profiler agent |
| `sample_text` | `str` | Text representation of the first N rows |
| `profile_data` | `dict` | Structured profile (shape, columns, missing values) |

## Output

| Key | Type | Description |
|---|---|---|
| `analyst_output` | `str` | Formatted markdown with grouped insights |
| `insights` | `list[dict]` | Structured insight objects |
| `error` | `str` (optional) | Error message if generation failed |

## Insight Structure

```json
{
  "title": "High concentration in top product category",
  "observation": "Category A represents 68% of all sales",
  "hypothesis": "Marketing budget is disproportionately allocated to Category A",
  "recommendation": "Analyze ROI per category and diversify if warranted",
  "priority": "high",
  "category": "distribution"
}
```

### Priority Levels

| Priority | Icon | Definition |
|---|---|---|
| `high` | 🔴 | Directly affects revenue, cost, or key business decisions |
| `medium` | 🟡 | Useful for optimization or deeper investigation |
| `low` | 🟢 | Informational, nice-to-know |

### Categories

| Category | Icon | Description |
|---|---|---|
| `trend` | 📈 | Temporal evolution, upward/downward movement |
| `anomaly` | ⚠️ | Outlier, unexpected value |
| `correlation` | 🔗 | Relationship between variables |
| `distribution` | 📊 | Spread, concentration, skewness |
| `general` | 💡 | Does not fit other categories |

## Categorization

Two modes available:

1. **Keyword-based** (default, fast): Scans insight text for category-specific keywords. No LLM call needed.
2. **LLM-based** (optional, accurate): Sends each insight to the LLM with the `categorizer` prompt from `config/prompts.yaml`.

```python
# Keyword mode (default)
agent = AnalystAgent(use_llm_categorization=False)

# LLM mode
agent = AnalystAgent(use_llm_categorization=True)
```

## Prompts

Located in `config/prompts.yaml` under the `analyst` and `categorizer` keys:

- **analyst.system**: Instructions for generating 3–6 structured JSON insights
- **analyst.human**: Template with `{profiler_output}`, `{rows}`, `{cols}`, `{missing_values}`, `{sample_text}`
- **categorizer.system**: Classification instructions for the 5 categories
- **categorizer.human**: Template with `{insight_json}`

## Streamlit UI (Insights Tab)

The Insights tab in `app/main.py` provides:

- **Summary KPIs**: Total insights, high/medium/low priority counts
- **Category breakdown** with icons
- **Two display modes**:
  - **Cards**: Expandable cards grouped by category
  - **Markdown**: Raw formatted markdown
- **Prerequisite**: The Profiler AI analysis must be generated first (used as input)

## Testing

```bash
# Unit tests only (no LLM / no Ollama needed)
uv run python tests/test_analyst.py

# Full test suite
uv run pytest tests/ -v
```

### Unit Tests (no LLM)

| Test | What it verifies |
|---|---|
| `test_extract_json` | JSON extraction from noisy LLM output |
| `test_validate_insight` | Required field validation |
| `test_normalize_priority` | Priority normalization (haute→high, etc.) |
| `test_categorize_by_keywords` | Keyword-based categorization |
| `test_fallback_parse_markdown` | Markdown fallback parsing |
| `test_insights_to_markdown` | Markdown formatting output |

### Integration Test (requires Ollama)

| Test | What it verifies |
|---|---|
| `test_with_ollama` | Full pipeline with mock profiler data → LLM → structured insights |

## Configuration

The analyst uses the **text model** configured in `.env`:

```env
OLLAMA_TEXT_MODEL=mistral    # Each contributor sets their own model
LLM_PROVIDER=ollama          # or "gemini"
```

## Files Modified/Created

| File | Action |
|---|---|
| `agents/analyst.py` | Created — core analyst agent |
| `agents/mock_profiler.py` | Created — mock data for testing |
| `config/prompts.yaml` | Modified — added analyst + categorizer prompts |
| `app/main.py` | Modified — added Insights tab (P2 UI) |
| `tests/test_analyst.py` | Created — unit + integration tests |
| `utils/llm.py` | Modified — unified LLMClient + functional API, load_dotenv, langchain_ollama |
