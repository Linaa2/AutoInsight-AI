"""Visualization pipeline package for AutoInsight-AI.

Public API
----------
The primary entry point for other modules (orchestrator, Streamlit UI) is:

    from agents.visualizer import generate_visualizations, run_visualization_pipeline

All contracts (dataclasses) are importable from :mod:`visualization.schemas`.
Lower-level utilities are also re-exported here for convenience.
"""

from visualization.executor import execute_chart
from visualization.parser import parse_llm_output
from visualization.schemas import (
    ALLOWED_CHART_TYPES,
    ChartExecutionResult,
    ChartSpec,
    RenderedChart,
    VisualizationPipelineResult,
    VisualizerLLMOutput,
    VisualizerRequest,
)

__all__ = [
    "ALLOWED_CHART_TYPES",
    "ChartExecutionResult",
    "ChartSpec",
    "RenderedChart",
    "VisualizationPipelineResult",
    "VisualizerLLMOutput",
    "VisualizerRequest",
    "execute_chart",
    "parse_llm_output",
]
