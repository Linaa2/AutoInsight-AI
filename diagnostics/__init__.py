"""Diagnostics & telemetry package for AutoInsight-AI."""

from diagnostics.telemetry import (
    build_telemetry,
    build_telemetry_no_llm,
    collect_resource_snapshot,
    get_model_info,
)


def render_diagnostics_tab(*args, **kwargs):
    """Lazily import the Streamlit renderer only when needed."""
    from diagnostics.renderer import render_diagnostics_tab as _render_diagnostics_tab

    return _render_diagnostics_tab(*args, **kwargs)


def render_pipeline_diagram(*args, **kwargs):
    """Lazily import the Streamlit renderer only when needed."""
    from diagnostics.renderer import render_pipeline_diagram as _render_pipeline_diagram

    return _render_pipeline_diagram(*args, **kwargs)


__all__ = [
    "build_telemetry",
    "build_telemetry_no_llm",
    "collect_resource_snapshot",
    "get_model_info",
    "render_diagnostics_tab",
    "render_pipeline_diagram",
]
