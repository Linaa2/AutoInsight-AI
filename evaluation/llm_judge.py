"""LLM-as-judge evaluation agent for AutoInsight-AI.

Architecture:
    - truncate_for_judge()        : Section-aware context-window management.
    - build_judge_system_prompt() : Compose the judge system prompt from rubric.
    - build_judge_human_prompt()  : Compose the judge human prompt with artifact.
    - parse_judge_response()      : Parse LLM JSON -> EvaluationResult (+ fallback).
    - EvaluationAgent             : Orchestrates evaluation for each artifact type.

All tuneable constants (truncation limits, judge model) are read from
evaluation.config and can be overridden via environment variables.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from evaluation.config import (
    EVAL_JUDGE_MODEL,
    EVAL_MAX_INSIGHT_FIELD_CHARS,
    EVAL_MAX_SECTION_CHARS,
)
from evaluation.rubrics import RUBRICS
from evaluation.schemas import (
    AnalystEvaluationResult,
    EvaluationCriterion,
    EvaluationResult,
    InsightScore,
    make_timestamp,
)
from evaluation.validators import (
    check_column_references,
    check_insight_fields,
    check_required_sections,
)
from utils.llm import call_llm_with_messages

if TYPE_CHECKING:
    from tools.profiler_engine import DataProfile

logger = logging.getLogger(__name__)

# Module-level aliases so callers can import these names from llm_judge
# (e.g. tests that verify truncation boundaries).
MAX_SECTION_CHARS: int = EVAL_MAX_SECTION_CHARS
MAX_INSIGHT_FIELD_CHARS: int = EVAL_MAX_INSIGHT_FIELD_CHARS


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------


def _load_prompts() -> dict:
    candidates = [
        Path(__file__).resolve().parent.parent / "config" / "prompts.yaml",
        Path.cwd() / "config" / "prompts.yaml",
    ]
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                result: dict = yaml.safe_load(f)
                return result
    raise FileNotFoundError(f"prompts.yaml not found. Searched: {[str(p) for p in candidates]}")


_PROMPTS: dict = _load_prompts()


# ---------------------------------------------------------------------------
# truncate_for_judge
# ---------------------------------------------------------------------------


def truncate_for_judge(artifact: str, artifact_type: str) -> str:
    """Truncate an artifact to fit in the judge prompt without losing coverage.

    Strategy:
    - Markdown artifacts (profiler, reporter): extract each ## section heading,
      keep up to EVAL_MAX_SECTION_CHARS of its body, then reassemble. Every
      section is represented, preventing the judge from seeing only the
      beginning of the document.
    - Analyst artifacts (JSON string): parse the insight array, truncate each
      insight's text fields to EVAL_MAX_INSIGHT_FIELD_CHARS, re-serialize.

    Args:
        artifact:      The raw artifact string.
        artifact_type: "profiler" | "analyst" | "reporter".

    Returns:
        A truncated string safe to embed in the judge prompt.
    """
    if not artifact:
        return artifact

    if artifact_type == "analyst":
        return _truncate_insights_json(artifact)
    return _truncate_markdown_sections(artifact)


def _truncate_markdown_sections(text: str) -> str:
    """Extract ## sections and cap each body at MAX_SECTION_CHARS."""
    parts = re.split(r"(?m)^(#{1,3} .+)$", text)

    if len(parts) <= 1:
        return text[: MAX_SECTION_CHARS * 6]

    result: list[str] = []

    if parts[0].strip():
        result.append(parts[0].strip()[:MAX_SECTION_CHARS])

    i = 1
    while i < len(parts) - 1:
        heading = parts[i]
        body = parts[i + 1].strip()
        truncated_body = body[:MAX_SECTION_CHARS]
        if len(body) > MAX_SECTION_CHARS:
            truncated_body += "..."
        result.append(f"{heading}\n{truncated_body}")
        i += 2

    return "\n\n".join(result)


def _truncate_insights_json(json_str: str) -> str:
    """Truncate text fields inside each insight dict."""
    _TEXT_FIELDS = ("title", "observation", "hypothesis", "recommendation")
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return json_str[: MAX_SECTION_CHARS * 6]

    if isinstance(data, list):
        insights = data
    elif isinstance(data, dict) and "insights" in data:
        insights = data["insights"]
    else:
        return json_str[: MAX_SECTION_CHARS * 6]

    truncated: list[dict] = []
    for ins in insights:
        t: dict[str, Any] = {}
        for k, v in ins.items():
            if k in _TEXT_FIELDS and isinstance(v, str) and len(v) > MAX_INSIGHT_FIELD_CHARS:
                t[k] = v[:MAX_INSIGHT_FIELD_CHARS] + "..."
            else:
                t[k] = v
        truncated.append(t)

    return json.dumps(truncated, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_judge_system_prompt(rubric: list[dict]) -> str:
    """Return the judge system prompt with criteria names embedded."""
    base = _PROMPTS["evaluation"]["judge"]["system"]
    criteria_names = ", ".join(c["name"] for c in rubric)
    return f"{base.rstrip()}\n\nExpected criteria keys: {criteria_names}"


def build_judge_human_prompt(
    artifact: str,
    validation_report: dict,
    rubric: list[dict],
    artifact_type: str,
    context: str = "",
) -> str:
    """Return the judge human prompt populated with artifact and metadata."""
    rubric_json = json.dumps(
        [{"name": c["name"], "label": c["label"], "weight": c["weight"]} for c in rubric],
        indent=2,
    )
    validation_json = json.dumps(validation_report, indent=2, default=str)

    if artifact_type == "analyst":
        template = _PROMPTS["evaluation"]["judge"]["analyst_human"]
        return template.format(
            rubric_json=rubric_json,
            validation_report_json=validation_json,
            artifact=artifact,
        )

    template = _PROMPTS["evaluation"]["judge"]["human"]
    return template.format(
        artifact_type=artifact_type,
        rubric_json=rubric_json,
        validation_report_json=validation_json,
        context=context or "(none)",
        artifact=artifact,
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _extract_json_from_response(raw: str) -> dict | None:
    """Strip markdown fences and extract the outermost JSON object."""
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
                    return None
    return None


def _fallback_result(artifact_type: str, raw: str) -> EvaluationResult:
    """Return a zero-score EvaluationResult when parsing fails."""
    return EvaluationResult(
        artifact_type=artifact_type,
        overall_score=0.0,
        grade="poor",
        criteria=[],
        critique=f"[Evaluation failed - raw model output]: {raw[:500]}",
        suggestions=["Re-run the evaluation with a more capable judge model."],
        judge_model=EVAL_JUDGE_MODEL,
        timestamp=make_timestamp(),
    )


def parse_judge_response(raw: str, artifact_type: str) -> EvaluationResult:
    """Parse the LLM judge JSON response into a typed EvaluationResult.

    Falls back to a zero-score result if JSON is malformed or incomplete.
    For artifact_type == "analyst", returns an AnalystEvaluationResult with
    per_insight populated.
    """
    data = _extract_json_from_response(raw)
    if data is None:
        logger.warning("Judge response could not be parsed as JSON (artifact=%s)", artifact_type)
        return _fallback_result(artifact_type, raw)

    rubric = RUBRICS.get(artifact_type, [])

    criteria: list[EvaluationCriterion] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for entry in rubric:
        name = entry["name"]
        label = entry["label"]
        weight = entry["weight"]
        raw_crit = data.get(name, {})

        if isinstance(raw_crit, dict):
            score = float(raw_crit.get("score", 0.0))
            rationale = str(raw_crit.get("rationale", ""))
        else:
            score = 0.0
            rationale = ""

        score = max(0.0, min(1.0, score))
        criteria.append(
            EvaluationCriterion(name=name, label=label, score=score, rationale=rationale)
        )
        weighted_sum += score * weight
        total_weight += weight

    overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    overall_score = round(max(0.0, min(1.0, overall_score)), 4)

    critique = str(data.get("overall_critique", ""))
    raw_suggestions = data.get("suggestions", [])
    suggestions = [str(s) for s in raw_suggestions] if isinstance(raw_suggestions, list) else []

    base_kwargs = {
        "artifact_type": artifact_type,
        "overall_score": overall_score,
        "grade": EvaluationResult.compute_grade(overall_score),
        "criteria": criteria,
        "critique": critique,
        "suggestions": suggestions,
        "judge_model": EVAL_JUDGE_MODEL,
        "timestamp": make_timestamp(),
    }

    if artifact_type != "analyst":
        return EvaluationResult(**base_kwargs)

    raw_per_insight = data.get("per_insight", [])
    per_insight: list[InsightScore] = []

    if isinstance(raw_per_insight, list):
        for item in raw_per_insight:
            if not isinstance(item, dict):
                continue
            per_insight.append(
                InsightScore(
                    title=str(item.get("title", "")),
                    factual_correctness=float(item.get("factual_correctness", 0.0)),
                    relevance=float(item.get("relevance", 0.0)),
                    actionability=float(item.get("actionability", 0.0)),
                    note=str(item.get("note", "")),
                )
            )

    return AnalystEvaluationResult(**base_kwargs, per_insight=per_insight)


# ---------------------------------------------------------------------------
# EvaluationAgent
# ---------------------------------------------------------------------------


class EvaluationAgent:
    """LLM-as-judge agent that evaluates AutoInsight-AI pipeline artifacts.

    Uses EVAL_JUDGE_MODEL (default: qwen3:14b) — a text-reasoning model — to
    assess quality across rubric criteria. Deterministic validators pre-compute
    ground-truth facts injected into the judge prompt.

    All configuration is read from evaluation.config and can be overridden via
    environment variables.

    Usage::

        agent = EvaluationAgent()
        result = agent.evaluate_profiler(profiler_output, profile)
        result = agent.evaluate_analyst(insights, profile)
        result = agent.evaluate_reporter(reporter_output, analyst_output)
    """

    def evaluate_profiler(
        self,
        profiler_output: str,
        profile: DataProfile,
    ) -> EvaluationResult:
        """Evaluate ProfilerAgent markdown output.

        Args:
            profiler_output: Markdown string produced by ProfilerAgent.
            profile:         DataProfile used as ground truth for validators.
        """
        validation_report = {
            "column_check": check_column_references(profiler_output, profile),
            "section_check": check_required_sections(profiler_output, "profiler"),
        }
        artifact = truncate_for_judge(profiler_output, "profiler")
        return self._run_judge("profiler", artifact, validation_report)

    def evaluate_analyst(
        self,
        insights: list[dict],
        profile: DataProfile,
    ) -> AnalystEvaluationResult:
        """Evaluate AnalystAgent insights in a single LLM call.

        Returns aggregate scores + per-insight breakdown in one
        AnalystEvaluationResult (no extra calls).

        Args:
            insights: List of insight dicts from AnalystAgent.
            profile:  DataProfile used as ground truth for validators.
        """
        validation_report = {
            "field_check": check_insight_fields(insights),
            "column_check": check_column_references(json.dumps(insights), profile),
        }
        artifact = truncate_for_judge(json.dumps(insights, ensure_ascii=False), "analyst")
        result = self._run_judge("analyst", artifact, validation_report)
        if not isinstance(result, AnalystEvaluationResult):
            return AnalystEvaluationResult(
                **{k: getattr(result, k) for k in vars(result)},
                per_insight=[],
            )
        return result

    def evaluate_reporter(
        self,
        reporter_output: str,
        analyst_output: str,
    ) -> EvaluationResult:
        """Evaluate ReporterAgent markdown output.

        Args:
            reporter_output: Markdown string produced by ReporterAgent.
            analyst_output:  Analyst markdown (used as faithfulness context).
        """
        validation_report = {
            "section_check": check_required_sections(reporter_output, "reporter"),
        }
        artifact = truncate_for_judge(reporter_output, "reporter")
        context = truncate_for_judge(analyst_output, "reporter")
        return self._run_judge("reporter", artifact, validation_report, context=context)

    def _run_judge(
        self,
        artifact_type: str,
        artifact: str,
        validation_report: dict,
        context: str = "",
    ) -> EvaluationResult:
        rubric = RUBRICS[artifact_type]
        system_prompt = build_judge_system_prompt(rubric)
        human_prompt = build_judge_human_prompt(
            artifact=artifact,
            validation_report=validation_report,
            rubric=rubric,
            artifact_type=artifact_type,
            context=context,
        )
        try:
            raw = call_llm_with_messages(system_prompt, human_prompt, EVAL_JUDGE_MODEL)
        except RuntimeError as exc:
            logger.error("Judge LLM call failed for %s: %s", artifact_type, exc)
            return _fallback_result(artifact_type, str(exc))

        return parse_judge_response(raw, artifact_type)
