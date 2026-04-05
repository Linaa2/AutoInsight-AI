"""Data model for the LLM-as-Judge evaluation subsystem.

Classes
-------
EvaluationCriterion      — score + rationale for a single rubric criterion.
EvaluationResult         — full evaluation of one pipeline artifact.
InsightScore             — per-insight breakdown (analyst only).
AnalystEvaluationResult  — EvaluationResult extended with per_insight list.
PipelineEvaluation       — aggregated evaluation across all three artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.config import (
    EVAL_EXCELLENT_THRESHOLD,
    EVAL_FAIR_THRESHOLD,
    EVAL_GOOD_THRESHOLD,
)


@dataclass
class EvaluationCriterion:
    """Score and rationale for a single rubric criterion."""

    name: str  # e.g. "grounding"
    label: str  # "Grounding"
    score: float  # 0.0 - 1.0
    rationale: str  # one sentence explaining the score
    weight: float  # contribution to overall score (all weights within artifact sum to 1.0)


@dataclass
class EvaluationResult:
    """Full quality evaluation of a single pipeline artifact."""

    artifact_type: str  # "profiler" | "analyst" | "reporter"
    overall_score: float  # weighted composite 0.0 - 1.0
    grade: str  # "excellent" | "good" | "fair" | "poor"
    criteria: list[EvaluationCriterion]
    critique: str  # 2-3 sentence overall assessment from the judge
    suggestions: list[str]  # 2-4 concrete improvement suggestions
    judge_model: str  # model that produced this result
    timestamp: str  # ISO-8601

    @staticmethod
    def compute_grade(score: float) -> str:
        """Map a composite score to a grade label using env-var thresholds."""
        if score >= EVAL_EXCELLENT_THRESHOLD:
            return "excellent"
        if score >= EVAL_GOOD_THRESHOLD:
            return "good"
        if score >= EVAL_FAIR_THRESHOLD:
            return "fair"
        return "poor"


@dataclass
class InsightScore:
    """Per-insight breakdown produced by the analyst judge call."""

    title: str
    factual_correctness: float  # 0.0 - 1.0
    relevance: float  # 0.0 - 1.0
    actionability: float  # 0.0 - 1.0
    note: str  # one-sentence per-insight note from the judge


@dataclass
class AnalystEvaluationResult(EvaluationResult):
    """Analyst evaluation — extends EvaluationResult with per-insight scores."""

    per_insight: list[InsightScore] = field(default_factory=list)


@dataclass
class PipelineEvaluation:
    """Aggregated evaluation across all three pipeline artifacts."""

    profiler_eval: EvaluationResult | None = None
    analyst_eval: AnalystEvaluationResult | None = None
    reporter_eval: EvaluationResult | None = None

    @property
    def pipeline_score(self) -> float:
        """Mean overall_score across whichever artifacts have been evaluated."""
        scores = [
            e.overall_score
            for e in (self.profiler_eval, self.analyst_eval, self.reporter_eval)
            if e is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def pipeline_grade(self) -> str:
        """Grade derived from pipeline_score using the same thresholds."""
        return EvaluationResult.compute_grade(self.pipeline_score)
