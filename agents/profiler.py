"""Profiler agent — generates a structured markdown description of a dataset profile."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.llm import LLMClient
from utils.prompt_loader import load_prompt_section

if TYPE_CHECKING:
    from tools.profiler_engine import DataProfile


class ProfilerAgent:
    """Uses an LLM to produce a human-readable markdown report from a :class:`DataProfile`.

    Prompts are loaded from ``config/prompts.yaml`` via the shared prompt loader.
    The LLM used is the **text** model (``OLLAMA_TEXT_MODEL``).
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()
        self._prompts = load_prompt_section("profiler")

    def describe(self, profile: DataProfile, callbacks: list | None = None) -> str:
        """Generate a structured markdown description for *profile*.

        Args:
            profile:   The deterministic profile produced by :class:`DataProfiler`.
            callbacks: Optional LangChain callbacks (e.g. LangFuse handler).

        Returns:
            Markdown-formatted analysis string.
        """
        profile_json = json.dumps(profile.to_dict(), indent=2, default=str)

        llm = self._llm_client.get_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._prompts["system"]),
                ("human", self._prompts["human"]),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"profile_json": profile_json}, config={"callbacks": callbacks or []})
