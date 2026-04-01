"""LLM-as-Judge evaluation agent (P10).

Evaluates the output *quality* of three pipeline artifacts:
    profiler  — Is the profile report accurate, complete, and readable?
    analyst   — Are the insights diverse, correct, and worth acting on?
    reporter  — Does the report faithfully and coherently tell the story?

Architecture
------------
- One LLM call per artifact (3 total for a full run).
- Deterministic validators (from evaluation.validators) inject ground truth
  into each judge prompt so the LLM confirms rather than re-discovers.
- Analyst call receives optional context from Critic and Uncertainty outputs.
- All results are typed dataclass instances — never crashes on LLM failure.

Public API
----------
    agent = EvaluationAgent()
    result = agent.evaluate_profiler(profile_markdown, profile_data)
    result = agent.evaluate_analyst(insights, profile_markdown, profile_data,
                                    critiques=..., confidence_scores=...)
    result = agent.evaluate_reporter(report_markdown, profile_markdown,
                                     insights_markdown)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any, cast

from evaluation.config import (
    EVAL_JUDGE_MODEL,
    EVAL_JUDGE_TIMEOUT,
    EVAL_MAX_INSIGHT_FIELD_CHARS,
    EVAL_MAX_SECTION_CHARS,
)
from evaluation.rubrics import RUBRICS
from evaluation.schemas import (
    AnalystEvaluationResult,
    EvaluationCriterion,
    EvaluationResult,
    InsightScore,
)
from evaluation.validators import (
    check_column_references,
    check_insight_fields,
    check_required_sections,
    check_statistic_accuracy,
)
from utils.llm import call_llm_with_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Criterion descriptions used when building system prompts
# ---------------------------------------------------------------------------

_CRITERIA_DESC: dict[str, dict[str, str]] = {
    "profiler": {
        "grounding": "Column names and statistics are correctly cited; no hallucinated values.",
        "completeness": "All required sections present; all column types covered.",
        "clarity": "Writing is clear, well-structured, avoids unexplained jargon.",
        "specificity": "Findings reference concrete numbers, not vague generalisations.",
    },
    "analyst": {
        "factual_correctness": "Claims are consistent with profile statistics.",
        "relevance": "Insights address meaningful business questions, not trivia.",
        "actionability": "Recommendations are concrete and implementable.",
        "priority_calibration": "High-priority flags are proportional to actual business impact.",
        "diversity": "Insights cover different aspects; no near-duplicate observations.",
    },
    "reporter": {
        "faithfulness": "All key findings from profiler and analyst are represented.",
        "coherence": "Sections flow logically; the narrative is consistent.",
        "language": "Accessible to a non-technical reader; no raw jargon.",
        "completeness": "All required sections present and substantive.",
        "actionability": "Recommendations are concrete and prioritised.",
    },
}

_ARTIFACT_ROLES: dict[str, str] = {
    "profiler": "dataset profiling report",
    "analyst": "data analyst insights report",
    "reporter": "executive analysis report",
}

# ---------------------------------------------------------------------------
# JSON schema templates embedded in system prompts
# ---------------------------------------------------------------------------

_PROFILER_REPORTER_SCHEMA = """{
  "criteria": {
    "<criterion_name>": {"score": <float 0-1>, "rationale": "<one sentence>"}
  },
  "critique":    "<2-3 sentence overall assessment>",
  "suggestions": ["<concrete improvement 1>", "<concrete improvement 2>"]
}"""

_ANALYST_SCHEMA = """{
  "criteria": {
    "<criterion_name>": {"score": <float 0-1>, "rationale": "<one sentence>"}
  },
  "critique":    "<2-3 sentence overall assessment>",
  "suggestions": ["<concrete improvement 1>"],
  "per_insight": [
    {
      "title": "<insight title>",
      "factual_correctness": <float 0-1>,
      "relevance":           <float 0-1>,
      "actionability":       <float 0-1>,
      "note":                "<one sentence>"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from raw LLM output (may include markdown fences)."""
    text = raw.strip()

    # 1. Try the raw string first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            result = json.loads(fenced.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 3. Find the outermost { ... }
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            result = json.loads(brace.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot extract JSON from LLM output: {raw[:300]!r}")


def _compute_weighted_score(criteria: list[EvaluationCriterion]) -> float:
    """Compute the weighted average score across all criteria."""
    total_weight = sum(c.weight for c in criteria)
    if total_weight == 0:
        return 0.5
    return sum(c.score * c.weight for c in criteria) / total_weight


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Section-aware truncation
# ---------------------------------------------------------------------------


def truncate_for_judge(
    text: str,
    max_chars_per_section: int = EVAL_MAX_SECTION_CHARS,
) -> str:
    """Truncate a markdown document per section so all headings remain visible.

    Each ``##`` (or ``###``) section body is capped to *max_chars_per_section*
    characters. Short documents are returned unchanged.

    Args:
        text:                  The markdown text to truncate.
        max_chars_per_section: Maximum characters per section body.

    Returns:
        Truncated string where each section body is at most
        *max_chars_per_section* characters.
    """
    # Split on any heading line (##, ###, ####, …)
    heading_pattern = re.compile(r"(^#{1,6}\s+.+$)", re.MULTILINE)
    parts = heading_pattern.split(text)

    # No headings — cap total length at 10x max_chars_per_section
    if len(parts) <= 1:
        limit = max_chars_per_section * 10
        return text if len(text) <= limit else text[:limit] + "\n[...truncated...]"

    result_parts: list[str] = []

    # Text before the first heading
    preamble = parts[0]
    if preamble.strip():
        result_parts.append(
            preamble
            if len(preamble) <= max_chars_per_section
            else preamble[:max_chars_per_section] + "\n[...truncated...]"
        )

    # Process heading + body pairs
    i = 1
    while i < len(parts):
        heading = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        result_parts.append(heading)
        if len(body) > max_chars_per_section:
            result_parts.append(body[:max_chars_per_section] + "\n[...truncated...]")
        else:
            result_parts.append(body)
        i += 2

    return "\n".join(result_parts)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_system_prompt(artifact_type: str) -> str:
    """Build the judge system prompt including rubric definitions and JSON schema."""
    rubric = RUBRICS[artifact_type]
    criteria_desc = _CRITERIA_DESC[artifact_type]
    role = _ARTIFACT_ROLES[artifact_type]
    schema = _ANALYST_SCHEMA if artifact_type == "analyst" else _PROFILER_REPORTER_SCHEMA

    rubric_lines = "\n".join(
        f"- {c['name']} ({int(float(c['weight']) * 100)}%): {criteria_desc[str(c['name'])]}"
        for c in rubric
    )

    return f"""You are an expert output quality judge for data analysis pipelines.
Your task is to evaluate the quality of a {role}.

RUBRIC:
{rubric_lines}

SCORING RULES:
- Score each criterion from 0.0 (very poor) to 1.0 (excellent).
- Use any validator output injected in the user prompt as ground truth — do NOT re-derive it.
- Criteria with insufficient content to evaluate receive 0.5 (neutral).
- Write 2-3 sentences in "critique" and 2-4 items in "suggestions".
- For the analyst artifact: include a "per_insight" list with one entry per insight.

REPLY ONLY with valid JSON matching this exact schema (no markdown, no backticks, no extra text):
{schema}"""


def _build_human_prompt(
    artifact_type: str,
    content: str,
    validator_results: dict[str, Any],
    context_hints: dict[str, str] | None = None,
) -> str:
    """Build the judge human prompt with validator ground truth and artifact content."""
    parts: list[str] = []

    # ── Optional context (analyst only) ───────────────────────────────────────
    if context_hints and artifact_type == "analyst":
        parts.append("## Prior Analysis Signals")
        parts.append(
            "Use the following as context only — "
            "do NOT substitute it for your own independent assessment.\n"
        )
        if context_hints.get("critic"):
            parts.append("### Critic Flags")
            parts.append(context_hints["critic"] + "\n")
        if context_hints.get("uncertainty"):
            parts.append("### Data Confidence Signals")
            parts.append(context_hints["uncertainty"] + "\n")

    # ── Validator results ──────────────────────────────────────────────────────
    parts.append("## Validator Results\n")

    if artifact_type in ("profiler", "reporter"):
        present = validator_results.get("present_sections", [])
        missing = validator_results.get("missing_sections", [])
        parts.append(
            f"Required sections present: {', '.join(present) if present else 'none'}\n"
            f"Missing sections: {', '.join(missing) if missing else 'none'}\n"
        )

    col_refs: dict[str, Any] = validator_results.get("column_refs", {})
    if col_refs:
        present_cols: list[str] = col_refs.get("present", [])
        unmentioned: list[str] = col_refs.get("unmentioned", [])
        parts.append(
            f"Columns referenced in text: "
            f"{', '.join(present_cols[:10]) if present_cols else 'none'}\n"
            f"Columns unmentioned: "
            f"{', '.join(unmentioned[:10]) if unmentioned else 'none'}\n"
        )

    stat_acc: dict[str, Any] = validator_results.get("stat_accuracy", {})
    if stat_acc:
        gt_lines: list[str] = []
        verified_set: set[str] = set(stat_acc.get("verified", []))
        for item in stat_acc.get("ground_truth", []):
            check = "verified" if item["stat"] in verified_set else "not found in text"
            gt_lines.append(f"  - {item['stat']}: {item['value']} ({check})")
        if gt_lines:
            parts.append("Ground truth statistics:\n" + "\n".join(gt_lines) + "\n")

    if artifact_type == "analyst":
        field_check: dict[str, Any] = validator_results.get("insight_fields", {})
        if field_check:
            valid = field_check.get("valid_count", 0)
            total = field_check.get("total", 0)
            issues: list[str] = field_check.get("issues", [])
            parts.append(
                f"Insight field validation: {valid}/{total} insights have all required fields."
            )
            if issues:
                parts.append("Issues: " + "; ".join(issues) + "\n")

    # ── Artifact content ───────────────────────────────────────────────────────
    artifact_label = {
        "profiler": "Profiler Report",
        "analyst": "Analyst Insights",
        "reporter": "Final Report",
    }.get(artifact_type, artifact_type.capitalize())

    parts.append(f"\n## {artifact_label}\n")
    parts.append(content)

    parts.append("\nEvaluate the artifact against the rubric. Return ONLY valid JSON.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Context builders (critic and uncertainty → analyst judge prompt)
# ---------------------------------------------------------------------------


def _build_critic_context(critiques: list[dict[str, Any]]) -> str:
    """Format critic review as a compact bullet list for the judge prompt."""
    if not critiques:
        return "No critic review available."
    lines: list[str] = []
    for c in critiques:
        title = c.get("insight_title", "Insight")
        verdict = c.get("verdict", "unknown")
        weakness = c.get("weaknesses", "—")
        lines.append(f'- "{title}" — verdict: {verdict}. Weakness: {weakness}')
    return "\n".join(lines)


def _build_uncertainty_context(confidence_scores: list[dict[str, Any]]) -> str:
    """Format confidence scores as a compact bullet list for the judge prompt."""
    if not confidence_scores:
        return "No confidence scores available."
    lines: list[str] = []
    for s in confidence_scores:
        title = s.get("insight_title", "Insight")
        score = s.get("confidence_score", 0)
        level = s.get("confidence_level", "unknown")
        summary = s.get("summary", "")
        lines.append(f'- "{title}" — {level} confidence ({score}%). {summary}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser and fallback
# ---------------------------------------------------------------------------


def _fallback_result(
    artifact_type: str,
    reason: str,
    *,
    judge_model: str = EVAL_JUDGE_MODEL,
) -> EvaluationResult:
    """Return a neutral EvaluationResult when the LLM call fails or returns garbage."""
    rubric = RUBRICS.get(artifact_type, [])
    criteria = [
        EvaluationCriterion(
            name=str(c["name"]),
            label=str(c["label"]),
            score=0.5,
            rationale=f"Could not evaluate: {reason[:120]}",
            weight=float(c["weight"]),
        )
        for c in rubric
    ]
    base_kwargs: dict[str, Any] = {
        "artifact_type": artifact_type,
        "overall_score": 0.5,
        "grade": "fair",
        "criteria": criteria,
        "critique": f"Evaluation unavailable — {reason[:200]}",
        "suggestions": ["Re-run evaluation when the LLM is available."],
        "judge_model": judge_model,
        "timestamp": _now_iso(),
    }
    if artifact_type == "analyst":
        return AnalystEvaluationResult(**base_kwargs, per_insight=[])
    return EvaluationResult(**base_kwargs)


def parse_judge_response(
    raw_json: str,
    artifact_type: str,
    *,
    judge_model: str = EVAL_JUDGE_MODEL,
) -> EvaluationResult:
    """Parse an LLM judge response into a typed EvaluationResult.

    Missing criteria are filled with defaults (score=0.5, rationale="Not evaluated").
    If JSON cannot be extracted, returns a fallback result.

    Args:
        raw_json:      Raw string from the LLM (may include markdown fences).
        artifact_type: ``"profiler"`` | ``"analyst"`` | ``"reporter"``

    Returns:
        ``EvaluationResult`` (or ``AnalystEvaluationResult`` for analyst).
    """
    rubric = RUBRICS.get(artifact_type, [])
    timestamp = _now_iso()

    try:
        data = _extract_json(raw_json)
    except ValueError as exc:
        logger.warning("parse_judge_response: %s", exc)
        return _fallback_result(
            artifact_type,
            f"JSON parse failed: {raw_json[:80]}",
            judge_model=judge_model,
        )

    criteria_raw: dict[str, Any] = data.get("criteria", {})
    criteria: list[EvaluationCriterion] = []

    for c in rubric:
        name = str(c["name"])
        label = str(c["label"])
        weight = float(c["weight"])
        c_data: dict[str, Any] = criteria_raw.get(name, {})

        raw_score = c_data.get("score", 0.5)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.5
        score = max(0.0, min(1.0, score))

        rationale = str(c_data.get("rationale", "Not evaluated by judge."))
        criteria.append(
            EvaluationCriterion(
                name=name, label=label, score=score, rationale=rationale, weight=weight
            )
        )

    overall_score = _compute_weighted_score(criteria)
    grade = EvaluationResult.compute_grade(overall_score)
    critique = str(data.get("critique", "No overall critique provided."))
    suggestions = [str(s) for s in data.get("suggestions", [])]

    if artifact_type == "analyst":
        per_insight_raw: list[dict[str, Any]] = data.get("per_insight", [])
        per_insight: list[InsightScore] = []
        for item in per_insight_raw:
            per_insight.append(
                InsightScore(
                    title=str(item.get("title", "Untitled")),
                    factual_correctness=_safe_float(item.get("factual_correctness", 0.5)),
                    relevance=_safe_float(item.get("relevance", 0.5)),
                    actionability=_safe_float(item.get("actionability", 0.5)),
                    note=str(item.get("note", "")),
                )
            )
        return AnalystEvaluationResult(
            artifact_type=artifact_type,
            overall_score=overall_score,
            grade=grade,
            criteria=criteria,
            critique=critique,
            suggestions=suggestions,
            judge_model=judge_model,
            timestamp=timestamp,
            per_insight=per_insight,
        )

    return EvaluationResult(
        artifact_type=artifact_type,
        overall_score=overall_score,
        grade=grade,
        criteria=criteria,
        critique=critique,
        suggestions=suggestions,
        judge_model=judge_model,
        timestamp=timestamp,
    )


def _safe_float(value: Any, default: float = 0.5) -> float:
    """Convert *value* to float, clamped to [0.0, 1.0], falling back to *default*."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Main evaluation agent
# ---------------------------------------------------------------------------


class EvaluationAgent:
    """LLM-as-Judge agent that scores pipeline artifact quality.

    Uses one LLM call per artifact (3 total). Deterministic validators inject
    ground truth before each call. Results are fully typed — never crashes.

    Usage::

        agent = EvaluationAgent()
        profiler_eval  = agent.evaluate_profiler(profile_markdown, profile_data)
        analyst_eval   = agent.evaluate_analyst(insights, profile_markdown, profile_data)
        reporter_eval  = agent.evaluate_reporter(report_markdown, profile_markdown,
                                                  insights_markdown)
    """

    def __init__(
        self,
        *,
        judge_model: str = EVAL_JUDGE_MODEL,
        judge_timeout: int = EVAL_JUDGE_TIMEOUT,
    ) -> None:
        self._judge_model = judge_model
        self._judge_timeout = judge_timeout

    def _call_judge(self, system: str, human: str, callbacks: list[Any] | None = None) -> str:
        """Run one judge request using the evaluation-specific model budget."""
        return call_llm_with_messages(
            system=system,
            human=human,
            model=self._judge_model,
            callbacks=callbacks,
            timeout=self._judge_timeout,
        )

    def evaluate_profiler(
        self,
        profile_markdown: str,
        profile_data: dict[str, Any],
        callbacks: list[Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate the profiler output quality.

        Args:
            profile_markdown: Profiler markdown report.
            profile_data:     Raw profile dict from ``DataProfiler.to_dict()``.
            callbacks:        Optional LangChain callbacks (e.g. LangFuse handler).

        Returns:
            ``EvaluationResult`` for the profiler artifact.
        """
        section_check = check_required_sections(profile_markdown, "profiler")
        validator_results: dict[str, Any] = {
            "present_sections": section_check["present"],
            "missing_sections": section_check["missing"],
            "column_refs": check_column_references(profile_markdown, profile_data),
            "stat_accuracy": check_statistic_accuracy(profile_markdown, profile_data),
        }

        truncated = truncate_for_judge(profile_markdown)
        system = _build_system_prompt("profiler")
        human = _build_human_prompt("profiler", truncated, validator_results)

        try:
            raw = self._call_judge(system=system, human=human, callbacks=callbacks)
            logger.info("EvaluationAgent.evaluate_profiler: LLM returned %d chars", len(raw))
            return parse_judge_response(raw, "profiler", judge_model=self._judge_model)
        except Exception as exc:
            logger.error("EvaluationAgent.evaluate_profiler failed: %s", exc, exc_info=True)
            return _fallback_result("profiler", str(exc), judge_model=self._judge_model)

    def evaluate_analyst(
        self,
        insights: list[dict[str, Any]],
        profile_markdown: str,
        profile_data: dict[str, Any],
        critiques: list[dict[str, Any]] | None = None,
        confidence_scores: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
    ) -> AnalystEvaluationResult:
        """Evaluate the analyst insights quality.

        Args:
            insights:          List of insight dicts from the analyst agent.
            profile_markdown:  Profiler markdown (reference for the judge).
            profile_data:      Raw profile dict.
            critiques:         Optional CriticAgent output (injected as context).
            confidence_scores: Optional UncertaintyEstimator output (injected as context).
            callbacks:         Optional LangChain callbacks.

        Returns:
            ``AnalystEvaluationResult`` with aggregate scores and per-insight breakdown.
        """
        max_chars = EVAL_MAX_INSIGHT_FIELD_CHARS
        formatted_insights: list[str] = []
        for ins in insights:
            trimmed = {
                k: (v[:max_chars] + "..." if isinstance(v, str) and len(v) > max_chars else v)
                for k, v in ins.items()
            }
            formatted_insights.append(json.dumps(trimmed, ensure_ascii=False))
        insights_text = "\n\n".join(formatted_insights)

        validator_results: dict[str, Any] = {
            "insight_fields": check_insight_fields(insights),
            "column_refs": check_column_references(insights_text, profile_data),
            "stat_accuracy": check_statistic_accuracy(insights_text, profile_data),
        }

        context_hints: dict[str, str] = {}
        if critiques:
            context_hints["critic"] = _build_critic_context(critiques)
        if confidence_scores:
            context_hints["uncertainty"] = _build_uncertainty_context(confidence_scores)

        profile_ref = truncate_for_judge(profile_markdown)
        content = (
            f"**Dataset Profile (reference):**\n{profile_ref}\n\n"
            f"**Insights to Evaluate:**\n{insights_text}"
        )

        system = _build_system_prompt("analyst")
        human = _build_human_prompt("analyst", content, validator_results, context_hints)

        try:
            raw = self._call_judge(system=system, human=human, callbacks=callbacks)
            logger.info("EvaluationAgent.evaluate_analyst: LLM returned %d chars", len(raw))
            result = parse_judge_response(raw, "analyst", judge_model=self._judge_model)
            return cast("AnalystEvaluationResult", result)
        except Exception as exc:
            logger.error("EvaluationAgent.evaluate_analyst failed: %s", exc, exc_info=True)
            return cast(
                "AnalystEvaluationResult",
                _fallback_result("analyst", str(exc), judge_model=self._judge_model),
            )

    def evaluate_reporter(
        self,
        report_markdown: str,
        profile_markdown: str,
        insights_markdown: str,
        callbacks: list[Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate the reporter output quality.

        Args:
            report_markdown:   Full reporter markdown report.
            profile_markdown:  Profiler markdown (reference for faithfulness check).
            insights_markdown: Analyst insights markdown (reference).
            callbacks:         Optional LangChain callbacks.

        Returns:
            ``EvaluationResult`` for the reporter artifact.
        """
        section_check = check_required_sections(report_markdown, "reporter")
        validator_results: dict[str, Any] = {
            "present_sections": section_check["present"],
            "missing_sections": section_check["missing"],
        }

        truncated = truncate_for_judge(report_markdown)

        # Include profile and insights as short references for faithfulness scoring
        profile_ref = truncate_for_judge(profile_markdown) if profile_markdown else ""
        insights_ref = truncate_for_judge(insights_markdown) if insights_markdown else ""
        content_parts = [f"**Final Report:**\n{truncated}"]
        if profile_ref:
            content_parts.append(f"\n**Profile Reference (faithfulness check):**\n{profile_ref}")
        if insights_ref:
            content_parts.append(f"\n**Insights Reference (faithfulness check):**\n{insights_ref}")
        content = "\n".join(content_parts)

        system = _build_system_prompt("reporter")
        human = _build_human_prompt("reporter", content, validator_results)

        try:
            raw = self._call_judge(system=system, human=human, callbacks=callbacks)
            logger.info("EvaluationAgent.evaluate_reporter: LLM returned %d chars", len(raw))
            return parse_judge_response(raw, "reporter", judge_model=self._judge_model)
        except Exception as exc:
            logger.error("EvaluationAgent.evaluate_reporter failed: %s", exc, exc_info=True)
            return _fallback_result("reporter", str(exc), judge_model=self._judge_model)
