"""Visualization pipeline package for AutoInsight-AI.

Public API
----------
The entry point for other modules (orchestrator, Streamlit UI) is:

    from visualization import generate_visualizations

    output = generate_visualizations(df, profile_summary, insights_text, columns_info)

All types used in the contract are importable from :mod:`visualization.schemas`.
"""

from visualization.executor import execute_chart
from visualization.parser import parse_llm_output
from visualization.schemas import (
    ALLOWED_CHART_TYPES,
    ChartResult,
    ChartSpec,
    ExecutionResult,
    VisualizationOutput,
)

__all__ = [
    "ALLOWED_CHART_TYPES",
    "ChartResult",
    "ChartSpec",
    "ExecutionResult",
    "VisualizationOutput",
    "execute_chart",
    "parse_llm_output",
]
