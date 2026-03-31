"""agents/uncertainty.py — Phase 8: Uncertainty Estimator Agent.

Assigns a confidence score (0-100%) to each analyst insight using a hybrid
approach: two rule-based drivers (fast, deterministic) and two LLM-based
drivers (semantic judgment).

Architecture:
    - UncertaintyEstimator (class): Scores each insight across 4 drivers.
    - UncertaintyFormatter (class): Converts scores to markdown.
    - Helper functions: rule-based scorers, LLM scorer, JSON extractor.
    - uncertainty_node(): Standalone LangGraph node entry point.

Inputs (from LangGraph state):
    - insights       : list[dict]   — from AnalystAgent
    - critiques      : list[dict]   — from CriticAgent (optional; defaults used if absent)
    - profile_data   : dict         — from DataProfiler

Outputs:
    - confidence_scores : list[dict]  — one score dict per insight
    - uncertainty_output: str         — markdown summary table
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar

from evaluation.config import EVAL_UNCERTAINTY_HIGH_THRESHOLD, EVAL_UNCERTAINTY_MEDIUM_THRESHOLD
from utils.llm import call_llm_with_messages
from utils.prompt_loader import load_prompt_section

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level prompt loading
# ---------------------------------------------------------------------------

PROMPTS: dict[str, Any] = {"uncertainty": load_prompt_section("uncertainty")}

# ---------------------------------------------------------------------------
# Verdict / confidence → numeric score mappings (fallback when LLM unavailable)
# ---------------------------------------------------------------------------

_VERDICT_SCORES: dict[str, int] = {
    "supported": 20,
    "partially_supported": 12,
    "weak": 5,
}

_CONFIDENCE_SCORES: dict[str, int] = {
    "high": 20,
    "medium": 12,
    "low": 5,
}

# Default score used when no critique AND LLM unavailable
_DEFAULT_CRITIC_SCORE = 12
_DEFAULT_LLM_SCORE = 12


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a (possibly noisy) LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    result: dict[str, Any] = json.loads(cleaned[start : i + 1])
                    return result
                except json.JSONDecodeError:
                    break
    return None


# ---------------------------------------------------------------------------
# Rule-based driver 1 — Data Quality (0-25 pts)
# ---------------------------------------------------------------------------


def _compute_data_quality(profile_data: dict[str, Any]) -> tuple[int, str]:
    """Score the dataset's structural quality. No LLM needed.

    Scoring table (max 25):
      Row count:  >=1000 -> 10 | 300-999 -> 7 | 100-299 -> 4 | <100 -> 1
      Missing %:  <5%    -> 10 | 5-20%  -> 6  | 20-40%  -> 3 | >40% -> 0
      Duplicates: <1%    ->  5 | 1-10%  -> 3  | >10%    -> 0
    """
    stats: dict[str, Any] = profile_data.get("stats", {})
    shape: list[int] = profile_data.get("shape", [0, 0])

    rows: int = int(shape[0]) if shape else 0
    missing_pct: float = float(stats.get("total_missing_pct", 0.0))
    dup_pct: float = float(stats.get("duplicates_pct", 0.0))

    # Row count sub-score
    if rows >= 1000:
        row_pts = 10
    elif rows >= 300:
        row_pts = 7
    elif rows >= 100:
        row_pts = 4
    else:
        row_pts = 1

    # Missing value sub-score
    if missing_pct < 5.0:
        miss_pts = 10
    elif missing_pct < 20.0:
        miss_pts = 6
    elif missing_pct < 40.0:
        miss_pts = 3
    else:
        miss_pts = 0

    # Duplicate sub-score
    if dup_pct < 1.0:
        dup_pts = 5
    elif dup_pct <= 10.0:
        dup_pts = 3
    else:
        dup_pts = 0

    score = row_pts + miss_pts + dup_pts
    reason = f"{rows} rows, {missing_pct:.1f}% missing, {dup_pct:.1f}% duplicates"
    return score, reason


# ---------------------------------------------------------------------------
# Rule-based driver 2 — Specificity (0-25 pts)
# ---------------------------------------------------------------------------


def _compute_specificity(
    insight: dict[str, Any],
    profile_data: dict[str, Any],
) -> tuple[int, str]:
    """Score how concrete and verifiable the insight text is. No LLM needed.

    Scoring table (max 25):
      >=1 referenced column name  -> +8
      >=2 referenced column names -> +4 (cumulative; cap 12)
      >=1 number / percentage     -> +8
      >=2 numbers / percentages   -> +3 (cumulative; cap 11)
      Actionable recommendation   -> +2
    """
    col_names: list[str] = list((profile_data.get("columns") or {}).keys())

    observation: str = str(insight.get("observation", ""))
    hypothesis: str = str(insight.get("hypothesis", ""))
    recommendation: str = str(insight.get("recommendation", ""))
    combined_text = f"{observation} {hypothesis}".lower()

    # Column reference count
    referenced_cols = [c for c in col_names if c.lower() in combined_text]
    n_cols = len(referenced_cols)
    if n_cols >= 2:
        col_pts = 12
    elif n_cols == 1:
        col_pts = 8
    else:
        col_pts = 0

    # Number / percentage count
    numbers = re.findall(r"\d+(?:\.\d+)?%?", combined_text)
    n_nums = len(numbers)
    if n_nums >= 2:
        num_pts = 11
    elif n_nums == 1:
        num_pts = 8
    else:
        num_pts = 0

    # Actionable recommendation (>=8 words, not generic)
    _weak_phrases = {"investigate", "further", "more analysis", "look into"}
    rec_words = recommendation.lower().split()
    is_weak = any(ph in recommendation.lower() for ph in _weak_phrases)
    actionable = len(rec_words) >= 8 and not is_weak
    rec_pts = 2 if actionable else 0

    score = col_pts + num_pts + rec_pts

    parts: list[str] = []
    if referenced_cols:
        parts.append(f"references {n_cols} column(s): {', '.join(referenced_cols[:3])}")
    if n_nums:
        parts.append(f"{n_nums} concrete number(s)")
    if rec_pts:
        parts.append("actionable recommendation")
    reason = "; ".join(parts) if parts else "no concrete details found"

    return score, reason


# ---------------------------------------------------------------------------
# LLM-based drivers 3 & 4 — Statistical Evidence + Critic Assessment (0-25 each)
# ---------------------------------------------------------------------------


def _build_profile_digest(
    insight: dict[str, Any],
    profile_data: dict[str, Any],
) -> str:
    """Return a compact profile summary for the LLM, scoped to referenced columns."""
    col_names: list[str] = list((profile_data.get("columns") or {}).keys())
    observation = str(insight.get("observation", ""))
    hypothesis = str(insight.get("hypothesis", ""))
    text = f"{observation} {hypothesis}".lower()

    referenced = [c for c in col_names if c.lower() in text] or col_names[:5]
    columns_data: dict[str, Any] = profile_data.get("columns") or {}
    stats: dict[str, Any] = profile_data.get("stats") or {}
    shape: list[int] = profile_data.get("shape", [0, 0])

    lines = [
        f"Dataset: {shape[0]} rows x {shape[1]} cols",
        f"Missing: {stats.get('total_missing_pct', 0):.1f}%",
        f"Duplicates: {stats.get('duplicates_pct', 0):.1f}%",
        "",
        "Referenced columns:",
    ]
    for col in referenced:
        col_info: dict[str, Any] = columns_data.get(col, {})
        mean_val = col_info.get("mean")
        std_val = col_info.get("std")
        miss_val = col_info.get("missing_pct", 0)
        dtype_cat = col_info.get("dtype_category", "unknown")
        stat_str = ""
        if mean_val is not None and std_val is not None:
            stat_str = f", mean={mean_val:.2f}, std={std_val:.2f}"
        lines.append(f"  - {col} ({dtype_cat}, {miss_val:.1f}% missing{stat_str})")

    return "\n".join(lines)


def _compute_llm_scores(
    insight: dict[str, Any],
    critique: dict[str, Any],
    profile_data: dict[str, Any],
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    """Call the LLM once to score both statistical_evidence and critic_assessment.

    Falls back gracefully if the LLM fails or the critique is empty.

    Returns:
        {
            "statistical_evidence": {"score": int, "reason": str},
            "critic_assessment":    {"score": int, "reason": str},
        }
    """
    # Determine fallback for critic_assessment before LLM call
    if not critique:
        critic_fallback_score = _DEFAULT_CRITIC_SCORE
        critic_fallback_reason = "No critique available — default score applied"
    else:
        verdict = str(critique.get("verdict", "")).lower()
        confidence = str(critique.get("confidence", "")).lower()
        critic_fallback_score = (
            _VERDICT_SCORES.get(verdict)
            or _CONFIDENCE_SCORES.get(confidence)
            or _DEFAULT_CRITIC_SCORE
        )
        critic_fallback_reason = (
            f"Critic verdict='{verdict}', confidence='{confidence}' (LLM unavailable)"
        )

    stat_fallback = {"score": _DEFAULT_LLM_SCORE, "reason": "LLM unavailable — default score"}
    critic_fallback = {"score": critic_fallback_score, "reason": critic_fallback_reason}

    try:
        profile_digest = _build_profile_digest(insight, profile_data)
        insight_json = json.dumps(
            {
                k: insight.get(k, "")
                for k in ("title", "observation", "hypothesis", "recommendation")
            },
            ensure_ascii=False,
        )
        critique_json = json.dumps(critique, ensure_ascii=False) if critique else "{}"

        system_prompt: str = PROMPTS["uncertainty"]["system"]
        human_prompt: str = PROMPTS["uncertainty"]["human"].format(
            insight_json=insight_json,
            profile_digest=profile_digest,
            critique_json=critique_json,
        )

        raw = call_llm_with_messages(
            system=system_prompt,
            human=human_prompt,
            callbacks=callbacks,
        )

        parsed = _extract_json(raw)
        if not parsed:
            logger.warning("UncertaintyEstimator: LLM returned unparseable response")
            return {"statistical_evidence": stat_fallback, "critic_assessment": critic_fallback}

        def _extract_driver(key: str, fallback: dict[str, Any]) -> dict[str, Any]:
            raw_val = parsed.get(key, {})
            if not isinstance(raw_val, dict):
                return fallback
            try:
                s = max(0, min(25, int(raw_val.get("score", fallback["score"]))))
                r = str(raw_val.get("reason", fallback["reason"]))
                return {"score": s, "reason": r}
            except (ValueError, TypeError):
                return fallback

        return {
            "statistical_evidence": _extract_driver("statistical_evidence", stat_fallback),
            "critic_assessment": _extract_driver("critic_assessment", critic_fallback),
        }

    except Exception as exc:
        logger.warning(f"UncertaintyEstimator: LLM call failed — {exc}")
        return {"statistical_evidence": stat_fallback, "critic_assessment": critic_fallback}


# ---------------------------------------------------------------------------
# Confidence level helper
# ---------------------------------------------------------------------------


def _determine_level(score: int) -> str:
    """Map a 0-100 integer score to a confidence level string."""
    if score >= EVAL_UNCERTAINTY_HIGH_THRESHOLD:
        return "high"
    if score >= EVAL_UNCERTAINTY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# UncertaintyEstimator
# ---------------------------------------------------------------------------


class UncertaintyEstimator:
    """Scores each analyst insight across four drivers to produce a 0-100% confidence score.

    Usage::

        estimator = UncertaintyEstimator()
        result = estimator.estimate_all(insights, critiques, profile_data)
        # result["confidence_scores"]  -> list[dict]
        # result["uncertainty_output"] -> markdown str
    """

    def estimate(
        self,
        insight: dict[str, Any],
        critique: dict[str, Any],
        profile_data: dict[str, Any],
        callbacks: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Compute a full confidence score for one insight.

        Always returns a valid dict — never raises.
        """
        try:
            dq_score, dq_reason = _compute_data_quality(profile_data)
            sp_score, sp_reason = _compute_specificity(insight, profile_data)
            llm_scores = _compute_llm_scores(insight, critique, profile_data, callbacks)

            stat_ev = llm_scores["statistical_evidence"]
            crit_as = llm_scores["critic_assessment"]

            total = dq_score + sp_score + stat_ev["score"] + crit_as["score"]
            level = _determine_level(total)
            title = str(insight.get("title", "Untitled"))

            return {
                "insight_title": title,
                "confidence_score": total,
                "confidence_level": level,
                "drivers": {
                    "data_quality": {"score": dq_score, "max": 25, "reason": dq_reason},
                    "specificity": {"score": sp_score, "max": 25, "reason": sp_reason},
                    "statistical_evidence": {
                        "score": stat_ev["score"],
                        "max": 25,
                        "reason": stat_ev["reason"],
                    },
                    "critic_assessment": {
                        "score": crit_as["score"],
                        "max": 25,
                        "reason": crit_as["reason"],
                    },
                },
                "summary": (
                    f"{level.capitalize()} confidence ({total}%). "
                    f"Data quality: {dq_score}/25 — Specificity: {sp_score}/25 — "
                    f"Statistical evidence: {stat_ev['score']}/25 — "
                    f"Critic assessment: {crit_as['score']}/25."
                ),
            }
        except Exception as exc:
            logger.error(f"UncertaintyEstimator.estimate failed: {exc}", exc_info=True)
            title = str(insight.get("title", "Untitled"))
            return {
                "insight_title": title,
                "confidence_score": 50,
                "confidence_level": "medium",
                "drivers": {
                    "data_quality": {"score": 12, "max": 25, "reason": "Error during scoring"},
                    "specificity": {"score": 12, "max": 25, "reason": "Error during scoring"},
                    "statistical_evidence": {
                        "score": 13,
                        "max": 25,
                        "reason": "Error during scoring",
                    },
                    "critic_assessment": {
                        "score": 13,
                        "max": 25,
                        "reason": "Error during scoring",
                    },
                },
                "summary": f"Scoring failed ({exc}); default 50% applied.",
            }

    def estimate_all(
        self,
        insights: list[dict[str, Any]],
        critiques: list[dict[str, Any]],
        profile_data: dict[str, Any],
        callbacks: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Score all insights and return the full result dict.

        Returns:
            {
                "confidence_scores":  list[dict],
                "uncertainty_output": str (markdown table),
            }
        """
        if not insights:
            return {"confidence_scores": [], "uncertainty_output": ""}

        critique_map: dict[str, dict[str, Any]] = {
            str(c.get("insight_title", "")): c for c in (critiques or [])
        }

        scores: list[dict[str, Any]] = []
        for insight in insights:
            title = str(insight.get("title", ""))
            matching_critique = critique_map.get(title, {})
            score = self.estimate(insight, matching_critique, profile_data, callbacks)
            scores.append(score)

        output = UncertaintyFormatter.to_markdown(scores)
        return {"confidence_scores": scores, "uncertainty_output": output}


# ---------------------------------------------------------------------------
# UncertaintyFormatter
# ---------------------------------------------------------------------------


class UncertaintyFormatter:
    """Converts a list of confidence score dicts to a markdown summary."""

    _LEVEL_ICONS: ClassVar[dict[str, str]] = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🔴",
    }

    @staticmethod
    def to_markdown(scores: list[dict[str, Any]]) -> str:
        """Build a compact markdown table from a list of confidence score dicts."""
        if not scores:
            return ""

        lines = [
            "## Confidence Scores\n",
            "| Insight | Score | Level | Data Quality | Specificity |"
            " Stat. Evidence | Critic Assessment |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in scores:
            icon = UncertaintyFormatter._LEVEL_ICONS.get(str(s.get("confidence_level", "")), "")
            level_str = f"{icon} {s.get('confidence_level', 'N/A')}"
            drivers: dict[str, Any] = s.get("drivers", {})
            dq = drivers.get("data_quality", {})
            sp = drivers.get("specificity", {})
            se = drivers.get("statistical_evidence", {})
            ca = drivers.get("critic_assessment", {})
            lines.append(
                f"| {s.get('insight_title', 'N/A')} "
                f"| {s.get('confidence_score', 0)}% "
                f"| {level_str} "
                f"| {dq.get('score', 0)}/25 "
                f"| {sp.get('score', 0)}/25 "
                f"| {se.get('score', 0)}/25 "
                f"| {ca.get('score', 0)}/25 |"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone LangGraph node
# ---------------------------------------------------------------------------


def uncertainty_node(state: dict[str, Any]) -> dict[str, Any]:
    """Standalone LangGraph node — compute uncertainty scores for all insights.

    Reads from state:
        - insights       (list[dict])
        - critiques      (list[dict], optional)
        - profile_data   (dict)

    Writes to state:
        - confidence_scores  (list[dict])
        - uncertainty_output (str)
    """
    insights: list[dict[str, Any]] = state.get("insights") or []
    critiques: list[dict[str, Any]] = state.get("critiques") or []
    profile_data: dict[str, Any] = state.get("profile_data") or {}

    if not insights:
        return {"confidence_scores": [], "uncertainty_output": ""}

    estimator = UncertaintyEstimator()
    result = estimator.estimate_all(insights, critiques, profile_data)
    return {
        "confidence_scores": result["confidence_scores"],
        "uncertainty_output": result["uncertainty_output"],
    }
