"""Evaluation data model for AutoInsight-AI.

Defines typed dataclasses for evaluation results produced by the LLM judge.
Grade thresholds are read from evaluation.config and can be overridden via
environment variables (EVAL_EXCELLENT_THRESHOLD, EVAL_GOOD_THRESHOLD,
EVAL_FAIR_THRESHOLD).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from evaluation.config import (
    EVAL_EXCELLENT_THRESHOLD,
    EVAL_FAIR_THRESHOLD,
    EVAL_GOOD_THRESHOLD,
)


def _grade(score: float) -> str:
    if score >= EVAL_EXCELLENT_THRESHOLD:
        return "excellent"
    if score >= EVAL_GOOD_THRESHOLD:
        return "good"
    if score >= EVAL_FAIR_THRESHOLD:
        return "fair"
    return "poor"


@dataclass
class EvaluationCriterion:
    """Score and rationale for a single rubric criterion."""

    name: str
    label: str
    score: float  # 0.0 - 1.0
    rationale: str


@dataclass
class EvaluationResult:
    """Full evaluation result for one agent artifact."""

    artifact_type: str  # "profiler" | "analyst" | "reporter"
    overall_score: float  # weighted average of criteria scores (0.0 - 1.0)
    grade: str  # "excellent" | "good" | "fair" | "poor"
    criteria: list[EvaluationCriterion]
    critique: str
    suggestions: list[str]
    judge_model: str
    timestamp: str

    @staticmethod
    def compute_grade(score: float) -> str:
        return _grade(score)


@dataclass
class InsightScore:
    """Per-insight scores returned inside an AnalystEvaluationResult."""

    title: str
    factual_correctness: float  # 0.0 - 1.0
    relevance: float
    actionability: float
    note: str


@dataclass
class AnalystEvaluationResult(EvaluationResult):
    """Analyst evaluation with additional per-insight breakdown."""

    per_insight: list[InsightScore] = field(default_factory=list)


@dataclass
class PipelineEvaluation:
    """Aggregate evaluation across all pipeline artifacts."""

    profiler_eval: EvaluationResult | None = None
    analyst_eval: AnalystEvaluationResult | None = None
    reporter_eval: EvaluationResult | None = None

    @property
    def pipeline_score(self) -> float:
        """Average overall_score across all non-None artifact evaluations."""
        scores = [
            e.overall_score
            for e in (self.profiler_eval, self.analyst_eval, self.reporter_eval)
            if e is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def pipeline_grade(self) -> str:
        return _grade(self.pipeline_score)


def make_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()
