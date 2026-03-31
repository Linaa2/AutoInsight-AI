"""Visualizer agent — the high-level entry point for the visualization pipeline.

Workflow
--------
1. :class:`VisualizerAgent` loads its prompt templates from
   ``config/prompts.yaml`` and builds a LangChain chain.
2. :func:`run_visualization_pipeline` calls the agent, parses the output, and
   executes each chart spec against the DataFrame.
3. :func:`generate_visualizations` is a backward-compatible wrapper that
   constructs a :class:`~visualization.schemas.VisualizerRequest` from the
   legacy positional arguments.

Integration contract
--------------------
Called by:
    - The LangGraph orchestrator (``graph/pipeline.py``)
    - The Streamlit UI (``app/pages/visualizer_playground.py``)

Input:
    A :class:`~visualization.schemas.VisualizerRequest` (or the legacy
    positional signature of :func:`generate_visualizations`).

Output:
    :class:`~visualization.schemas.VisualizationPipelineResult` — a dataclass
    with ``charts``, ``raw_llm_output``, and ``parsing_error`` fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.llm import LLMClient
from utils.prompt_loader import load_prompt_section
from visualization.executor import execute_chart
from visualization.parser import parse_llm_output
from visualization.schemas import (
    ALLOWED_CHART_TYPES,
    RenderedChart,
    VisualizationPipelineResult,
    VisualizerLLMOutput,
    VisualizerRequest,
)

if TYPE_CHECKING:
    import pandas as pd
    from langchain_core.language_models import BaseChatModel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _columns_info_from_df(df: pd.DataFrame) -> str:
    """Build a ``"col (dtype), ..."`` string from DataFrame column metadata."""
    return ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class VisualizerAgent:
    """LLM-driven chart-proposal agent.

    Loads prompt templates from ``config/prompts.yaml`` and wraps a LangChain
    chain that converts a :class:`~visualization.schemas.VisualizerRequest`
    into raw LLM text or a parsed :class:`~visualization.schemas.VisualizerLLMOutput`.

    Args:
        llm: Optional pre-built LangChain chat model.  When omitted,
             :meth:`~utils.llm.LLMClient.get_code_llm` is used to obtain the
             configured code-generation model (``OLLAMA_CODE_MODEL``).
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        prompts = load_prompt_section("visualizer")
        _llm = llm or LLMClient.get_code_llm()
        self._chain = (
            ChatPromptTemplate.from_messages(
                [("system", prompts["system"]), ("human", prompts["human"])]
            )
            | _llm
            | StrOutputParser()
        )

    def generate_raw(self, request: VisualizerRequest, callbacks: list | None = None) -> str:
        """Call the LLM and return the unmodified response string.

        Args:
            request:   The fully populated :class:`VisualizerRequest`.
            callbacks: Optional LangChain callbacks (e.g. LangFuse handler).

        Returns:
            The raw text returned by the LLM (expected to be JSON).
        """
        return self._chain.invoke(
            {
                "columns_info": request.columns_info,
                "profile_markdown": request.profile_markdown,
                "insights_markdown": request.insights_markdown,
                "allowed_chart_types": ", ".join(sorted(ALLOWED_CHART_TYPES)),
            },
            config={"callbacks": callbacks or []},
        )

    def generate(self, request: VisualizerRequest) -> VisualizerLLMOutput:
        """Call the LLM, parse the response, and return validated chart specs.

        Args:
            request: The fully populated :class:`VisualizerRequest`.

        Returns:
            A :class:`VisualizerLLMOutput` wrapping the valid chart specs.
            Invalid specs are silently dropped; call :meth:`generate_raw` if
            you need the full error picture.
        """
        raw = self.generate_raw(request)
        specs, _ = parse_llm_output(raw)
        return VisualizerLLMOutput(charts=specs)


# ---------------------------------------------------------------------------
# High-level pipeline function
# ---------------------------------------------------------------------------


def run_visualization_pipeline(
    df: pd.DataFrame,
    request: VisualizerRequest,
    llm: BaseChatModel | None = None,
    callbacks: list | None = None,
) -> VisualizationPipelineResult:
    """Run the full visualization pipeline and return structured results.

    Steps:
        1. Call the LLM via :class:`VisualizerAgent`.
        2. Parse and validate chart specs via :func:`~visualization.parser.parse_llm_output`.
        3. Execute each chart's code against *df* via
           :func:`~visualization.executor.execute_chart`.

    This function never raises.  All failures (LLM unreachable, bad JSON,
    code execution errors) are captured and surfaced in the return value.

    Args:
        df:        The dataset as a pandas DataFrame.
        request:   The pre-built :class:`VisualizerRequest`.
        llm:       Optional pre-built LangChain chat model (passed to the agent).
        callbacks: Optional LangChain callbacks (e.g. LangFuse handler).

    Returns:
        A :class:`VisualizationPipelineResult` with ``charts``,
        ``raw_llm_output``, and ``parsing_error`` fields.
    """
    agent = VisualizerAgent(llm=llm)

    try:
        raw_output = agent.generate_raw(request, callbacks=callbacks)
    except RuntimeError as exc:
        return VisualizationPipelineResult(
            charts=[],
            raw_llm_output="",
            parsing_error=f"LLM call failed: {exc}",
        )

    specs, parsing_error = parse_llm_output(raw_output)

    charts: list[RenderedChart] = [
        RenderedChart(spec=spec, execution=execute_chart(df, spec.code)) for spec in specs
    ]

    return VisualizationPipelineResult(
        charts=charts,
        raw_llm_output=raw_output,
        parsing_error=parsing_error,
    )


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------


def generate_visualizations(
    df: pd.DataFrame,
    profile_summary: str,
    insights_text: str,
    columns_info: str = "",
) -> VisualizationPipelineResult:
    """Backward-compatible wrapper around :func:`run_visualization_pipeline`.

    Accepts the legacy positional signature and constructs a
    :class:`VisualizerRequest` before delegating to the pipeline.

    Args:
        df:              The dataset as a pandas DataFrame.
        profile_summary: Text description of the dataset (from Profiler agent).
        insights_text:   Business insights text (from Analyst agent).
        columns_info:    Optional pre-formatted ``"col (dtype), ..."`` string.
                         Derived from ``df.dtypes`` when not provided.

    Returns:
        A :class:`VisualizationPipelineResult`.
    """
    resolved_columns_info = columns_info or _columns_info_from_df(df)
    request = VisualizerRequest(
        profile_markdown=profile_summary,
        insights_markdown=insights_text,
        columns_info=resolved_columns_info,
    )
    return run_visualization_pipeline(df, request)
