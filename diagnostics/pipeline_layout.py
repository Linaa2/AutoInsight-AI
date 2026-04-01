"""Shared pipeline layout metadata for Streamlit diagnostics views.

This module stays dependency-light so both the main app and diagnostics
renderer can import the canonical pipeline order without dragging in UI code.
"""

from __future__ import annotations

from orchestration.graph import _AGENTS_ORDER as _CONTENT_AGENT_ORDER

PIPELINE_NODE_ORDER = [*_CONTENT_AGENT_ORDER, "rag_storage"]
PIPELINE_ROWS = [
    ["start", "profiler", "analyst", "critic", "uncertainty"],
    ["visualizer", "reporter", "rag_storage", "end"],
]
PIPELINE_STATUS_CAPTION = (
    "Order follows arrows across two rows: "
    "START → Profiler → Analyst → Critic → Uncertainty → "
    "Visualizer → Reporter → Memory → END"
)
