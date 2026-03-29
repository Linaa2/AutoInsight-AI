"""Type contracts for the visualization pipeline.

All public types used between the visualizer agent, parser, and executor are
defined here so that the rest of the project can import from a single place.

Design note: plain TypedDicts are used instead of dataclasses or Pydantic to
keep dependencies zero and stay readable for students.
"""

from __future__ import annotations

from typing import Any, TypedDict

# The only chart types the LLM is permitted to propose.
# Kept small to avoid ambiguous or rarely-supported Plotly code.
ALLOWED_CHART_TYPES: frozenset[str] = frozenset(
    {"bar", "line", "scatter", "histogram", "box", "heatmap", "pie"}
)


class _ChartSpecRequired(TypedDict):
    """Required fields of a chart specification."""

    title: str  # human-readable chart title
    chart_type: str  # one of ALLOWED_CHART_TYPES
    code: str  # Python code that produces a Plotly figure named `fig`


class ChartSpec(_ChartSpecRequired, total=False):
    """Full chart spec returned by the LLM and validated by the parser.

    Required fields (always present):
        title, chart_type, code

    Optional fields (present when the LLM provides them):
        explanation    — one-sentence rationale for the chart choice
        columns_used   — list of DataFrame column names referenced in the code
    """

    explanation: str
    columns_used: list[str]


class ExecutionResult(TypedDict):
    """Result of executing one chart's code against a DataFrame.

    Attributes:
        success: True if the code ran and assigned a figure to `fig`.
        figure:  The Plotly Figure object when success is True, else None.
        error:   Human-readable error message when success is False, else None.
    """

    success: bool
    figure: Any | None  # plotly.graph_objects.Figure when successful
    error: str | None


class ChartResult(TypedDict):
    """A single chart: its validated spec paired with its execution result."""

    spec: ChartSpec
    execution: ExecutionResult


class VisualizationOutput(TypedDict):
    """Top-level object returned by ``generate_visualizations()``.

    This is the contract that the Streamlit UI and the LangGraph pipeline consume.

    Attributes:
        charts:          List of chart results (spec + execution status).
        raw_llm_output:  The unmodified text the LLM produced, kept for debugging.
        parsing_error:   Non-None when the LLM output could not be fully parsed.
    """

    charts: list[ChartResult]
    raw_llm_output: str
    parsing_error: str | None
