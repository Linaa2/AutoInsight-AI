"""Diagnostics & telemetry package for AutoInsight-AI."""

from diagnostics.renderer import render_diagnostics_tab, render_pipeline_diagram
from diagnostics.telemetry import (
    build_telemetry,
    build_telemetry_no_llm,
    collect_resource_snapshot,
    get_model_info,
)

__all__ = [
    "build_telemetry",
    "build_telemetry_no_llm",
    "collect_resource_snapshot",
    "get_model_info",
    "render_diagnostics_tab",
    "render_pipeline_diagram",
]
