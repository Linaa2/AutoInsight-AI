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

from agents.context_digest import build_profile_digest
from config.settings import settings
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
    """Extract a JSON object from a potentially noisy LLM response.

    Handles chain-of-thought reasoning blocks (<think>…</think>) emitted by
    models such as qwen3 before the actual JSON answer.
    """
    # Remove reasoning blocks first — they appear before the real answer and
    # contain their own curly braces that confuse the JSON extractor.
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
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
            profile_summary = build_profile_digest(profile_data)
            batch_size = max(1, settings.CRITIC_BATCH_SIZE)

            for start in range(0, len(insights), batch_size):
                chunk = insights[start : start + batch_size]
                critiques.extend(
                    self._critique_batch(
                        chunk,
                        profile_summary=profile_summary,
                        profile_data=profile_data,
                    )
                )

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
        *,
        profile_summary: str | None = None,
    ) -> dict:
        """Critique a single insight using the LLM.

        Always returns a critique dict — falls back to a graceful placeholder
        when the LLM response cannot be parsed, so the critic never silently
        drops an insight from its output.
        """
        title = insight.get("title", "Untitled")
        try:
            system_prompt = PROMPTS["system"]
            human_prompt = PROMPTS["human"].format(
                insight_json=json.dumps(insight, ensure_ascii=False, indent=2),
                profile_summary=profile_summary or build_profile_digest(profile_data),
            )

            raw = call_llm_with_messages(
                system=system_prompt,
                human=human_prompt,
                task="critic",
            )
            critique = self._parse_response(raw)

            if critique:
                critique["insight_title"] = title
                return critique

            # Parse failed — emit a fallback so this insight is still acknowledged.
            logger.warning("Critic could not parse LLM response for '%s'; using fallback.", title)
            return self._fallback_critique(title, reason="parse_error")

        except Exception as e:
            logger.warning("Failed to critique insight '%s': %s", title, e)
            return self._fallback_critique(title, reason=str(e))

    def _critique_batch(
        self,
        insights: list[dict],
        *,
        profile_summary: str,
        profile_data: dict | None = None,
    ) -> list[dict]:
        """Critique a chunk of insights in one LLM call, with safe fallback."""
        if not insights:
            return []

        requests_text = "\n\n".join(
            "\n".join(
                [
                    f"### Insight {index}",
                    "```json",
                    json.dumps(insight, ensure_ascii=False, indent=2),
                    "```",
                ]
            )
            for index, insight in enumerate(insights)
        )

        critiques_by_index: dict[int, dict] = {}
        try:
            raw = call_llm_with_messages(
                system=PROMPTS["system"],
                human=PROMPTS["batch_human"].format(
                    profile_summary=profile_summary,
                    requests_text=requests_text,
                ),
                task="critic",
            )
            critiques_by_index = self._parse_batch_response(raw)
        except Exception as exc:
            logger.warning("Critic batch call failed: %s", exc)

        critiques: list[dict] = []
        for index, insight in enumerate(insights):
            critique = critiques_by_index.get(index)
            if critique is not None:
                critique["insight_title"] = insight.get("title", "Untitled")
                critiques.append(critique)
                continue

            critiques.append(
                self._critique_single(
                    insight,
                    profile_data=profile_data,
                    profile_summary=profile_summary,
                )
            )

        return critiques

    @staticmethod
    def _fallback_critique(title: str, reason: str = "unknown") -> dict:
        """Return a conservative fallback critique when the LLM response is unusable."""
        return {
            "insight_title": title,
            "strengths": "Could not be assessed — LLM response was unparseable.",
            "weaknesses": f"Automatic review failed ({reason}). Manual review recommended.",
            "alternatives": "N/A",
            "confidence": "low",
            "verdict": "partially_supported",
        }

    def _parse_response(self, raw: str) -> dict | None:
        """Parse LLM response into a validated critique dict."""
        parsed = extract_json(raw)

        if parsed and validate_critique(parsed):
            parsed["verdict"] = normalize_verdict(parsed.get("verdict", ""))
            parsed["confidence"] = normalize_confidence(parsed.get("confidence", ""))
            return parsed

        logger.warning(f"Failed to parse critique. Raw:\n{raw[:300]}")
        return None

    def _parse_batch_response(self, raw: str) -> dict[int, dict]:
        """Parse a batched critique response keyed by chunk-local index."""
        parsed = extract_json(raw)
        if not parsed:
            logger.warning("Failed to parse batched critique response. Raw:\n%s", raw[:300])
            return {}

        items = parsed.get("critiques", [])
        if not isinstance(items, list):
            return {}

        critiques_by_index: dict[int, dict] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_index = item.get("index")
            if raw_index is None:
                continue
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue

            critique = {
                "strengths": item.get("strengths", ""),
                "weaknesses": item.get("weaknesses", ""),
                "alternatives": item.get("alternatives", ""),
                "confidence": item.get("confidence", ""),
                "verdict": item.get("verdict", ""),
            }
            if not validate_critique(critique):
                continue

            critique["verdict"] = normalize_verdict(str(critique["verdict"]))
            critique["confidence"] = normalize_confidence(str(critique["confidence"]))
            critiques_by_index[index] = critique

        return critiques_by_index


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
