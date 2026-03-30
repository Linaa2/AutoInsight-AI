"""Profiler agent — generates a structured markdown description of a dataset profile."""

import json
import os
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tools.profiler_engine import DataProfile
from utils.llm import LLMClient

_DEFAULT_PROMPTS_PATH = Path(__file__).parent.parent / "config" / "prompts.yaml"


class ProfilerAgent:
    """Uses an LLM to produce a human-readable markdown report from a :class:`DataProfile`.

    Environment variables:
        PROMPTS_PATH: Override path to the prompts YAML file.
                      Defaults to config/prompts.yaml relative to the project root.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm_client = llm_client or LLMClient()
        self._prompts = self._load_prompts()

    def describe(self, profile: DataProfile) -> str:
        """Generate a structured markdown description for *profile*.

        Args:
            profile: The deterministic profile produced by :class:`DataProfiler`.

        Returns:
            Markdown-formatted analysis string.
        """
        profile_json = json.dumps(profile.to_dict(), indent=2, default=str)

        llm = self._llm_client.get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._prompts["system"]),
            ("human", self._prompts["human"]),
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"profile_json": profile_json})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_prompts(self) -> dict:
        path = Path(os.getenv("PROMPTS_PATH", str(_DEFAULT_PROMPTS_PATH)))
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if "profiler" not in data:
            raise KeyError(f"'profiler' key not found in prompts file: {path}")
        return data["profiler"]
