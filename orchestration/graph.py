"""LangGraph orchestration graph for AutoInsight-AI.

This module defines the multi-agent workflow as a LangGraph ``StateGraph``.
Each agent is a thin **node function** that:

1. Reads inputs from :class:`~orchestration.state.PipelineState`.
2. Delegates to the corresponding agent class (``agents/``).
3. Writes outputs back into the state.

Current graph::

    START → profiler_node → visualizer_node → END

Future extensions (analyst, reporter, critic) can be added as new nodes
with edges inserted into the sequence or as conditional branches.

Usage::

    from orchestration.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "df_dict": df.to_dict(orient="records"),
        "file_name": "sales.csv",
    })
"""

from __future__ import annotations

import dataclasses

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.profiler import ProfilerAgent
from agents.visualizer import run_visualization_pipeline
from orchestration.state import PipelineState
from tools.profiler_engine import DataProfiler
from visualization.schemas import VisualizerRequest

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def profiler_node(state: PipelineState) -> PipelineState:
    """Run the deterministic profiler + LLM profiler agent.

    Reads:
        ``df_dict`` — the dataset as a list of row dicts.

    Writes:
        ``profile_data``     — raw profile as a dict.
        ``profile_markdown`` — LLM-generated markdown report.
        ``error``            — set on failure.
    """
    try:
        df = pd.DataFrame(state["df_dict"])
        profiler = DataProfiler()
        profile = profiler.profile(df)

        agent = ProfilerAgent()
        markdown = agent.describe(profile)

        return {
            "profile_data": profile.to_dict(),
            "profile_markdown": markdown,
        }
    except Exception as exc:
        return {"error": f"Profiler failed: {exc}"}


def visualizer_node(state: PipelineState) -> PipelineState:
    """Run the visualizer pipeline if profile + insights are available.

    Reads:
        ``df_dict``, ``profile_markdown``, ``insights_markdown``.

    Writes:
        ``visualization_result`` — serialised pipeline result (chart specs + metadata).
        ``error``                — set on failure.

    If ``insights_markdown`` is absent, the node uses the profile markdown
    as a stand-in so the visualizer is still exercisable during testing.
    """
    profile_md = state.get("profile_markdown")
    if not profile_md:
        return {"error": "Visualizer skipped: no profile_markdown available."}

    # Use analyst insights if present, otherwise fall back to profile text.
    insights = state.get("insights_markdown") or profile_md

    try:
        df = pd.DataFrame(state["df_dict"])
        columns_info = ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())
        request = VisualizerRequest(
            profile_markdown=profile_md,
            insights_markdown=insights,
            columns_info=columns_info,
        )

        result = run_visualization_pipeline(df, request)

        # Serialise for state storage (figures are not JSON-safe, store specs only).
        serialised: dict = {
            "raw_llm_output": result.raw_llm_output,
            "parsing_error": result.parsing_error,
            "charts": [
                {
                    "spec": dataclasses.asdict(rc.spec),
                    "execution": {
                        "success": rc.execution.success,
                        "error": rc.execution.error,
                    },
                }
                for rc in result.charts
            ],
        }
        return {"visualization_result": serialised}
    except Exception as exc:
        return {"error": f"Visualizer failed: {exc}"}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and compile the AutoInsight-AI orchestration graph.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``.invoke()``.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("profiler", profiler_node)
    graph.add_node("visualizer", visualizer_node)

    graph.add_edge(START, "profiler")
    graph.add_edge("profiler", "visualizer")
    graph.add_edge("visualizer", END)

    return graph.compile()
