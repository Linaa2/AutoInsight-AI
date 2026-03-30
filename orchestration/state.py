"""Shared graph state for the LangGraph orchestration pipeline.

The :class:`PipelineState` ``TypedDict`` is the contract between all graph
nodes.  Each node reads what it needs, writes its outputs, and leaves
everything else untouched.

Design choices
--------------
* **TypedDict** with ``total=False`` is used because LangGraph's
  ``StateGraph`` natively supports TypedDicts for key-based state updates.
* Fields anticipated by future agents (``insights_markdown``, ``report``,
  ``feedback``) are included but remain optional.
  This keeps the graph extensible without breaking existing nodes.
* ``df_dict`` (a ``list[dict]``) is used instead of a raw ``DataFrame``
  because LangGraph serialises state when checkpointing; a dict-list is
  JSON-safe.  Nodes reconstruct the ``DataFrame`` locally.
"""

from __future__ import annotations

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    """Shared state flowing through the LangGraph pipeline.

    All fields are optional (``total=False``) so that each node only returns
    the keys it produces, without needing to echo back the rest.

    Attributes:
        df_dict:                JSON-safe representation of the dataset
                                (``df.to_dict(orient="records")``).
                                Nodes convert back via ``pd.DataFrame(df_dict)``.
        file_name:              Original file name (for display / logging).
        profile_data:           Raw profile dict (from ``DataProfile.to_dict()``).
        profile_markdown:       Markdown report produced by the Profiler agent.
        insights_markdown:      Insights text produced by the Analyst agent (Phase 2).
        visualization_result:   Serialisable visualizer output (chart specs + metadata).
        report:                 Final report produced by the Reporter agent (future).
        feedback:               Critic feedback for self-correction loops (future).
        error:                  Non-``None`` when a node encounters an error.
    """

    # ---- required inputs ----
    df_dict: list[dict[str, Any]]
    file_name: str

    # ---- profiler outputs ----
    profile_data: dict[str, Any]
    profile_markdown: str

    # ---- analyst outputs (Phase 2) ----
    insights_markdown: str

    # ---- visualizer outputs ----
    visualization_result: dict[str, Any]

    # ---- reporter outputs (future) ----
    report: str

    # ---- critic / self-correction (future) ----
    feedback: str

    # ---- diagnostics ----
    error: str
