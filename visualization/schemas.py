"""Schema contracts for the visualization pipeline.

All data structures used between the visualizer agent, parser, and executor are
defined here so the rest of the project imports from a single place.

Design: standard-library ``dataclasses`` are used for clean, typed contracts
without adding new dependencies.  Fields are explicit, defaults are safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Allowed chart types
# ---------------------------------------------------------------------------

#: The only chart types the LLM is permitted to propose.
#: Kept small to prevent ambiguous or unsupported Plotly one-liners.
ALLOWED_CHART_TYPES: frozenset[str] = frozenset(
    {"bar", "line", "scatter", "histogram", "box", "heatmap", "pie"}
)


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


@dataclass
class VisualizerRequest:
    """Input contract for the VisualizerAgent.

    All context needed to propose charts is collected here, replacing the
    previous loose function arguments.

    Attributes:
        profile_markdown:  Text output of the Profiler agent.
        insights_markdown: Text output of the Analyst agent.
        columns_info:      Human-readable column list, e.g. ``"age (int64), name (object)"``.
        dataset_summary:   Optional one-paragraph dataset description.
    """

    profile_markdown: str
    insights_markdown: str
    columns_info: str
    dataset_summary: str | None = None


# ---------------------------------------------------------------------------
# LLM output contracts
# ---------------------------------------------------------------------------


@dataclass
class ChartSpec:
    """Specification for a single chart, as returned by the LLM.

    The ``code`` field contains executable Python that assigns a Plotly figure
    to the variable ``fig`` using ``df`` as the pre-loaded DataFrame.

    Attributes:
        title:        Human-readable chart title.
        chart_type:   One of :data:`ALLOWED_CHART_TYPES`.
        code:         Single-/two-line Plotly code assigning to ``fig``.
        explanation:  Optional one-sentence rationale for this chart choice.
        columns_used: DataFrame column names referenced in ``code``.
    """

    title: str
    chart_type: str
    code: str
    explanation: str | None = None
    columns_used: list[str] = field(default_factory=list)


@dataclass
class VisualizerLLMOutput:
    """Parsed output of the VisualizerAgent's LLM call.

    Wraps the validated list of ChartSpec objects produced by the parser.
    """

    charts: list[ChartSpec]


# ---------------------------------------------------------------------------
# Execution contracts
# ---------------------------------------------------------------------------


@dataclass
class ChartExecutionResult:
    """Result of executing one chart's code against a DataFrame.

    Attributes:
        success: ``True`` when the code ran and assigned a figure to ``fig``.
        error:   Human-readable error message on failure, ``None`` on success.
        figure:  The Plotly Figure object on success, ``None`` on failure.
    """

    success: bool
    error: str | None
    figure: Any | None  # plotly.graph_objects.Figure when successful


@dataclass
class RenderedChart:
    """A chart specification paired with the result of executing its code.

    This is the per-chart element inside :class:`VisualizationPipelineResult`.
    """

    spec: ChartSpec
    execution: ChartExecutionResult


# ---------------------------------------------------------------------------
# Top-level pipeline result
# ---------------------------------------------------------------------------


@dataclass
class VisualizationPipelineResult:
    """Top-level output of the full visualization pipeline.

    This is the contract consumed by the Streamlit UI and the LangGraph
    orchestrator.  It replaces the old ``VisualizationOutput`` TypedDict.

    Attributes:
        charts:          List of rendered charts (spec + execution status).
        raw_llm_output:  The unmodified LLM text, kept for debugging.
        parsing_error:   Non-``None`` when the LLM output could not be parsed.
    """

    charts: list[RenderedChart]
    raw_llm_output: str
    parsing_error: str | None
