"""
agents/critic.py — Phase 8: Critic Agent.

Reviews each insight from the Analyst, identifying strengths, weaknesses,
alternative hypotheses, and assigning a verdict.

Architecture:
    - CriticAgent (class): Reviews insights adversarially.
    - CriticFormatter (class): Formats critiques to markdown.
    - critic_node(): LangGraph node entry point.

Input (from LangGraph state):
    - insights        : list[dict]  (from Analyst)
    - profile_data    : dict        (structured profile)

Output:
    - critiques       : list[dict]  (structured critique per insight)
    - critic_output   : str         (formatted markdown)

Each critique is a dict:
    {
        "insight_title": str,
        "strengths": str,
        "weaknesses": str,
        "alternatives": str,
        "confidence": "high" | "medium" | "low",
        "verdict": "supported" | "partially_supported" | "weak"
    }
"""

from __future__ import annotations

import json
import logging
import re
from typing import ClassVar

from utils.llm import call_llm_with_messages
from utils.prompt_loader import load_prompt_section

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPT LOADER
# ═══════════════════════════════════════════════════════════════════════════


PROMPTS = load_prompt_section("critic")


# ═══════════════════════════════════════════════════════════════════════════
#  PARSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def extract_json(raw: str) -> dict | None:
    """Extract a JSON object from a potentially noisy LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.replace("```", "").strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    brace_count = 0
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                try:
                    result: dict = json.loads(cleaned[start : i + 1])
                    return result
                except json.JSONDecodeError:
                    break
    return None


def validate_critique(critique: dict) -> bool:
    """Check that a critique has all required fields."""
    required = {"strengths", "weaknesses", "alternatives", "confidence", "verdict"}
    return all(key in critique and critique[key] for key in required)


def normalize_verdict(verdict: str) -> str:
    """Normalize verdict to supported/partially_supported/weak."""
    v = verdict.strip().lower().replace(" ", "_")
    if v in ("supported", "strong", "valid"):
        return "supported"
    if v in ("weak", "unsupported", "invalid", "flawed"):
        return "weak"
    return "partially_supported"


def normalize_confidence(confidence: str) -> str:
    """Normalize confidence to high/medium/low."""
    c = confidence.strip().lower()
    if c in ("high", "haute", "strong"):
        return "high"
    if c in ("low", "basse", "weak", "faible"):
        return "low"
    return "medium"


# ═══════════════════════════════════════════════════════════════════════════
#  CRITIC FORMATTER
# ═══════════════════════════════════════════════════════════════════════════


class CriticFormatter:
    """Formats structured critiques into readable markdown."""

    VERDICT_ICONS: ClassVar[dict[str, str]] = {
        "supported": "✅",
        "partially_supported": "⚠️",
        "weak": "❌",
    }

    CONFIDENCE_ICONS: ClassVar[dict[str, str]] = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🔴",
    }

    @staticmethod
    def to_markdown(critiques: list[dict]) -> str:
        """Convert critiques to structured markdown."""
        if not critiques:
            return "⚠️ No critiques could be generated."

        lines = ["# 🔎 Insight Critiques\n"]

        for critique in critiques:
            verdict = critique.get("verdict", "partially_supported")
            confidence = critique.get("confidence", "medium")
            v_icon = CriticFormatter.VERDICT_ICONS.get(verdict, "⚠️")
            c_icon = CriticFormatter.CONFIDENCE_ICONS.get(confidence, "🟡")

            title = critique.get("insight_title", "Untitled Insight")
            lines.append(f"## {v_icon} {title}\n")
            lines.append(
                f"**Verdict**: {verdict.replace('_', ' ').capitalize()} | "
                f"**Confidence**: {c_icon} {confidence.capitalize()}\n"
            )
            lines.append(f"**Strengths**: {critique.get('strengths', 'N/A')}\n")
            lines.append(f"**Weaknesses**: {critique.get('weaknesses', 'N/A')}\n")
            lines.append(f"**Alternative hypotheses**: {critique.get('alternatives', 'N/A')}\n")
            lines.append("---\n")

        # Summary
        supported = sum(1 for c in critiques if c.get("verdict") == "supported")
        partial = sum(1 for c in critiques if c.get("verdict") == "partially_supported")
        weak = sum(1 for c in critiques if c.get("verdict") == "weak")
        lines.append(
            f"\n> **Summary**: {len(critiques)} insights reviewed — "
            f"✅ {supported} supported, ⚠️ {partial} partially supported, ❌ {weak} weak."
        )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  CRITIC AGENT
# ═══════════════════════════════════════════════════════════════════════════


class CriticAgent:
    """Adversarial agent that reviews and critiques analyst insights.

    For each insight, the Critic evaluates:
        - Strengths: what is well-supported
        - Weaknesses: logical gaps, unsupported assumptions
        - Alternatives: hypotheses the analyst may have missed
        - Confidence: how confident the critic is in the insight
        - Verdict: supported / partially_supported / weak

    Usage:
        critic = CriticAgent()
        result = critic.run(insights, profile_data)
    """

    def __init__(self) -> None:
        self.formatter = CriticFormatter()

    def run(
        self,
        insights: list[dict],
        profile_data: dict | None = None,
    ) -> dict:
        """
        Critique all insights.

        Args:
            insights: List of insight dicts from the Analyst.
            profile_data: Structured profile dict (optional).

        Returns:
            Dict with 'critiques' (list[dict]) and 'critic_output' (markdown).
        """
        try:
            critiques: list[dict] = []

            for insight in insights:
                critique = self._critique_single(insight, profile_data)
                if critique:
                    critiques.append(critique)

            markdown = self.formatter.to_markdown(critiques)

            logger.info(f"CriticAgent: {len(critiques)} critiques generated")

            return {
                "critiques": critiques,
                "critic_output": markdown,
            }

        except Exception as e:
            logger.error(f"Error in CriticAgent.run: {e}", exc_info=True)
            return {
                "critiques": [],
                "critic_output": f"❌ Error during critique: {e}",
                "error": str(e),
            }

    def _critique_single(
        self,
        insight: dict,
        profile_data: dict | None = None,
    ) -> dict | None:
        """Critique a single insight using the LLM."""
        try:
            system_prompt = PROMPTS["system"]
            human_prompt = PROMPTS["human"].format(
                insight_json=json.dumps(insight, ensure_ascii=False, indent=2),
                profile_summary=self._build_profile_summary(profile_data),
            )

            raw = call_llm_with_messages(system=system_prompt, human=human_prompt)
            critique = self._parse_response(raw)

            if critique:
                critique["insight_title"] = insight.get("title", "Untitled")

            return critique

        except Exception as e:
            logger.warning(f"Failed to critique insight '{insight.get('title', '?')}': {e}")
            return None

    def _parse_response(self, raw: str) -> dict | None:
        """Parse LLM response into a validated critique dict."""
        parsed = extract_json(raw)

        if parsed and validate_critique(parsed):
            parsed["verdict"] = normalize_verdict(parsed.get("verdict", ""))
            parsed["confidence"] = normalize_confidence(parsed.get("confidence", ""))
            return parsed

        logger.warning(f"Failed to parse critique. Raw:\n{raw[:300]}")
        return None

    def _build_profile_summary(self, profile_data: dict | None) -> str:
        """Build a compact profile summary for the LLM context."""
        if not profile_data:
            return "No structured profile available."

        shape = profile_data.get("shape", {})
        missing = profile_data.get("missing_values", {})
        duplicates = profile_data.get("duplicates", 0)

        columns = []
        for col in profile_data.get("columns", []):
            col_str = f"  - {col['name']} ({col['dtype']}): {col.get('unique', '?')} unique"
            if col.get("missing_pct", 0) > 0:
                col_str += f", {col['missing_pct']}% missing"
            columns.append(col_str)

        return (
            f"Rows: {shape.get('rows', '?')}, Columns: {shape.get('cols', '?')}\n"
            f"Duplicates: {duplicates}\n"
            f"Missing values: {json.dumps(missing, ensure_ascii=False)}\n"
            f"Columns:\n" + "\n".join(columns)
        )


# ═══════════════════════════════════════════════════════════════════════════
#  LANGGRAPH NODE
# ═══════════════════════════════════════════════════════════════════════════


def critic_node(state: dict) -> dict:
    """
    LangGraph node — Phase 8: critique analyst insights.

    Reads from state:
        - insights (list[dict])
        - profile_data (dict, optional)

    Writes to state:
        - critiques (list[dict])
        - critic_output (str)
    """
    insights = state.get("insights", [])
    profile_data = state.get("profile_data")

    if not insights:
        return {
            "critiques": [],
            "critic_output": "⚠️ No insights to critique.",
            "error": "No insights provided",
        }

    agent = CriticAgent()
    return agent.run(
        insights=insights,
        profile_data=profile_data,
    )
