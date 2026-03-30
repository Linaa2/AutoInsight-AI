# Phase 4: Reporter Agent

## Overview

The Reporter agent (`agents/reporter.py`) synthesizes outputs from the Profiler (P1), Analyst (P2), and Visualizer (P3) into a cohesive, executive-ready markdown report.

## Architecture

```
agents/reporter.py        → Core reporter agent + helpers + LangGraph node
config/prompts.yaml       → Reporter prompt (reporter section)
app/main.py               → Streamlit UI (Report tab)
tests/test_reporter.py    → Unit + integration tests
```

### Class

| Class | Role |
|---|---|
| `ReporterAgent` | Synthesizes all pipeline outputs into a structured markdown report |

### Helper Functions

| Function | Role |
|---|---|
| `extract_chart_titles()` | Extracts chart titles from visualizer output |
| `build_charts_summary()` | Formats chart titles into a bullet list for the prompt |
| `build_insights_summary()` | Formats insights with priority, category, and observations |
| `reporter_node()` | LangGraph node entry point |

## Pipeline

```
profiler_output + analyst_output + insights + visualizer_output
        │
        ▼
  ┌────────────────┐
  │  ReporterAgent  │
  │     .run()      │
  └───────┬─────────┘
          │
    1. build_charts_summary()    → Format chart titles as bullet list
    2. build_insights_summary()  → Format insights with priority/category
    3. Load prompts from config/prompts.yaml
    4. Call LLM with system + human messages
    5. Return {"reporter_output": str}
          │
          ▼
  { "reporter_output": str }
```

## Input

| Key | Type | Description |
|---|---|---|
| `profiler_output` | `str` | Markdown report from the Profiler agent |
| `analyst_output` | `str` | Markdown report from the Analyst agent |
| `insights` | `list[dict]` (optional) | Structured insight objects from the Analyst |
| `visualizer_output` | `dict` (optional) | Contains `"charts"` key with list of chart dicts |

## Output

| Key | Type | Description |
|---|---|---|
| `reporter_output` | `str` | Full executive-ready markdown report |
| `error` | `str` (optional) | Error message if generation failed |

## Report Sections

The generated report follows this structure:

| Section | Description |
|---|---|
| 📝 **Executive Summary** | 3–5 sentences for decision-makers: dataset size, number of insights, most critical finding |
| 📊 **Dataset Description** | Source, size, time period, data quality, missing values, duplicates |
| 🔍 **Key Insights** | Insights reformulated narratively, grouped by theme/priority, with business value |
| 📈 **Visualizations** | One sentence interpreting each chart; suggests charts if none were generated |
| ✅ **Recommendations** | 3–5 concrete, actionable items referencing specific insights, prioritized by impact |
| ⚠️ **Limitations & Next Steps** | What the analysis doesn't cover, 2–3 next steps, additional data sources |

## Prompts

Located in `config/prompts.yaml` under the `reporter` key:

- **reporter.system**: Instructions for writing a professional, non-technical executive report
- **reporter.human**: Template with `{profiler_output}`, `{analyst_output}`, `{insights_summary}`, `{charts_summary}`

## Streamlit UI (Report Tab)

The Report tab in `app/main.py` provides:

- **Prerequisites**: Requires both Profiler AI analysis and Analyst insights to be generated first
- **Generate button**: `📄 Generate Full Report`
- **Display**: Full markdown report rendered in the UI
- **Download**: `⬇️ Download Report` button saves the report as `analysis_report.md`

### Session State Keys

| Key | Source |
|---|---|
| `ai_description` | Profiler output (P1) |
| `analyst_markdown` | Analyst markdown (P2) |
| `analyst_insights` | Analyst structured insights (P2) |
| `reporter_output` | Reporter output (P3) |

## Testing

```bash
# Unit tests only (no LLM / no Ollama needed)
uv run python tests/test_reporter.py

# Full test suite
uv run pytest tests/ -v
```

### Unit Tests (no LLM)

| Test | What it verifies |
|---|---|
| `test_extract_chart_titles` | Extracts chart titles from visualizer dict |
| `test_extract_chart_titles_empty` | Handles None, empty dict, empty charts list |
| `test_build_charts_summary` | Formats chart titles as bullet list |
| `test_build_charts_summary_none` | Returns fallback message when no charts |
| `test_build_insights_summary` | Formats insights with priority, category, numbering |
| `test_build_insights_summary_empty` | Handles None and empty list |

### Integration Tests (require Ollama)

| Test | What it verifies |
|---|---|
| `test_reporter_agent_with_llm` | Full `ReporterAgent.run()` pipeline with mock data |
| `test_reporter_node_with_llm` | LangGraph `reporter_node()` with mock state |

## Configuration

The reporter uses the **text model** configured in `.env`:

```env
OLLAMA_TEXT_MODEL=mistral    # Each contributor sets their own model
LLM_PROVIDER=ollama          # or "gemini"
```

## Files Modified/Created

| File | Action |
|---|---|
| `agents/reporter.py` | Created — core reporter agent |
| `config/prompts.yaml` | Modified — added reporter prompts |
| `app/main.py` | Modified — added Report tab (P3 UI) with download button |
| `tests/test_reporter.py` | Created — unit + integration tests |
