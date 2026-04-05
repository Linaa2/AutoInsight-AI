"""Profiler agent — generates a structured markdown description of a dataset profile."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.profiler_context import build_profiler_prompt_context
from config.settings import settings
from utils.llm import LLMClient
from utils.prompt_loader import load_prompt_section

if TYPE_CHECKING:
    from tools.profiler_engine import DataProfile

logger = logging.getLogger(__name__)


class ProfilerAgent:
    """Uses an LLM to produce a human-readable markdown report from a :class:`DataProfile`.

    Prompts are loaded from ``config/prompts.yaml`` via the shared prompt loader.
    The LLM used is the **text** model (``OLLAMA_TEXT_MODEL``).

    The full deterministic :class:`~tools.profiler_engine.DataProfile` is kept intact;
    only a compact summary is sent to the LLM to reduce token usage and latency.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()
        self._prompts = load_prompt_section("profiler")

    def describe(
        self,
        profile: DataProfile,
        callbacks: list | None = None,
        detail_mode: str | None = None,
    ) -> str:
        """Generate a structured markdown description for *profile*.

        Args:
            profile:     The deterministic profile produced by :class:`DataProfiler`.
            callbacks:   Optional LangChain callbacks (e.g. LangFuse handler).
            detail_mode: ``"fast"`` or ``"full"``.  Falls back to
                         ``settings.PROFILER_DETAIL_MODE`` when *None*.

        Returns:
            Markdown-formatted analysis string.
        """
        t0 = time.perf_counter()

        mode = detail_mode if detail_mode in ("fast", "full") else settings.PROFILER_DETAIL_MODE

        # Build compact context — replaces the full profile.to_dict() JSON
        ctx = build_profiler_prompt_context(profile, mode)
        t_ctx = time.perf_counter()
        logger.debug(
            "Profiler context built in %.3fs | mode=%s | columns=%d | highlights=%d",
            t_ctx - t0,
            mode,
            len(ctx["column_summaries"]),
            len(ctx["highlights"]),
        )

        profile_summary_json = json.dumps(ctx, indent=2, default=str)

        llm = self._llm_client.get_task_llm("profiler")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._prompts["system"]),
                ("human", self._prompts["human"]),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke(
            {"profile_summary_json": profile_summary_json},
            config={"callbacks": callbacks or []},
        )

        t_llm = time.perf_counter()
        logger.debug(
            "Profiler LLM generation in %.3fs | total=%.3fs",
            t_llm - t_ctx,
            t_llm - t0,
        )

        return result
