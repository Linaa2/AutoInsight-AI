"""Visualizer agent — the high-level entry point for Phase 3.

Workflow
--------
1. ``generate_visualizations`` builds a prompt from the dataset context.
2. The prompt is sent to the local Ollama LLM via ``call_llm``.
3. The raw LLM response is parsed and validated by ``parse_llm_output``.
4. Each chart spec's code is executed against the DataFrame by ``execute_chart``.
5. A structured :class:`~visualization.schemas.VisualizationOutput` is returned.

Integration contract
--------------------
This function is meant to be called by:
    - The LangGraph orchestrator (``graph/pipeline.py``)
    - The Streamlit UI (``app/main.py``) directly for demo purposes

Input:
    df              — the uploaded dataset as a pandas DataFrame
    profile_summary — text summary produced by the Profiler agent
    insights_text   — insights produced by the Analyst agent
    columns_info    — optional pre-formatted "column (dtype)" string

Output: VisualizationOutput (see visualization/schemas.py for the full shape)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.llm import call_llm
from visualization.executor import execute_chart
from visualization.parser import parse_llm_output

if TYPE_CHECKING:
    import pandas as pd

    from visualization.schemas import ChartResult, VisualizationOutput

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# Uses str.format() placeholders ({columns_info} etc.) rather than an f-string
# because the JSON example inside the template contains literal { } characters
# that would require clunky escaping in an f-string.  {{ }} → { } after .format().

_PROMPT_TEMPLATE = """\
You are a data visualization assistant. Given a dataset profile and business \
insights, propose 3 to 5 charts that best help the user understand the data.

AVAILABLE COLUMNS (use ONLY these — never invent column names):
{columns_info}

DATASET PROFILE SUMMARY:
{profile_summary}

INSIGHTS TO ILLUSTRATE:
{insights_text}

STRICT RULES — follow exactly:
1. Output ONLY valid JSON.  Your response must start with {{ and end with }}.
   No markdown, no prose before or after the JSON.
2. Use only the column names listed above.  Do NOT invent new column names.
3. Each "code" value must use the variable `df` (a pandas DataFrame, already loaded).
4. The code must assign the final Plotly figure to a variable named `fig`.
5. Use only plotly.express (px) or plotly.graph_objects (go).  No import statements.
6. Allowed chart_type values: bar, line, scatter, histogram, box, heatmap, pie.
7. Keep each code snippet simple and directly executable (one or two lines).
8. Prefer charts that directly support the insights above.

REQUIRED OUTPUT FORMAT:
{{
  "charts": [
    {{
      "title": "Sales by Region",
      "chart_type": "bar",
      "code": "fig = px.bar(df, x='region', y='sales', title='Sales by Region')",
      "explanation": "Shows how total sales are distributed across regions.",
      "columns_used": ["region", "sales"]
    }}
  ]
}}\
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(profile_summary: str, insights_text: str, columns_info: str) -> str:
    """Render the prompt template with the caller-supplied context strings."""
    return _PROMPT_TEMPLATE.format(
        columns_info=columns_info,
        profile_summary=profile_summary,
        insights_text=insights_text,
    )


def _columns_info_from_df(df: pd.DataFrame) -> str:
    """Build a human-readable column list from a DataFrame.

    Produces a string like ``"region (object), sales (float64)"`` which the LLM
    can reference directly when writing column names in generated code.

    This is used automatically when the caller does not supply ``columns_info``.
    """
    return ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_visualizations(
    df: pd.DataFrame,
    profile_summary: str,
    insights_text: str,
    columns_info: str = "",
) -> VisualizationOutput:
    """Run the full visualization pipeline and return structured results.

    This is the single function the rest of the project calls.

    Args:
        df:              The dataset as a pandas DataFrame.
        profile_summary: Text description of the dataset (from Profiler agent).
        insights_text:   Business insights text (from Analyst agent).
        columns_info:    Optional pre-formatted column description string.
                         When empty, it is derived from ``df.dtypes`` automatically.

    Returns:
        A :class:`~visualization.schemas.VisualizationOutput` dict with:
            - ``charts``          — list of (spec, execution) pairs
            - ``raw_llm_output``  — unmodified LLM text (useful for debugging)
            - ``parsing_error``   — error description, or None on full success

    This function never raises.  All failures (LLM unreachable, bad JSON,
    code execution errors) are captured and surfaced in the return value.
    """
    # Fall back to deriving column info directly from the DataFrame schema
    resolved_columns_info = columns_info or _columns_info_from_df(df)

    prompt = _build_prompt(profile_summary, insights_text, resolved_columns_info)

    # --- Step 1: call the LLM ---
    try:
        raw_output = call_llm(prompt)
    except RuntimeError as exc:
        return {
            "charts": [],
            "raw_llm_output": "",
            "parsing_error": f"LLM call failed: {exc}",
        }

    # --- Step 2: parse and validate the LLM response ---
    specs, parsing_error = parse_llm_output(raw_output)

    # --- Step 3: execute each validated chart spec against the DataFrame ---
    chart_results: list[ChartResult] = [
        {"spec": spec, "execution": execute_chart(df, spec["code"])} for spec in specs
    ]

    return {
        "charts": chart_results,
        "raw_llm_output": raw_output,
        "parsing_error": parsing_error,
    }
