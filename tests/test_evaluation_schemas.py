"""Unit tests for evaluation/schemas.py."""

from evaluation.schemas import (
    AnalystEvaluationResult,
    EvaluationCriterion,
    EvaluationResult,
    InsightScore,
    PipelineEvaluation,
    make_timestamp,
)

# ---------------------------------------------------------------------------
# EvaluationCriterion
# ---------------------------------------------------------------------------


def test_evaluation_criterion_construction():
    c = EvaluationCriterion(
        name="grounding",
        label="Grounding",
        score=0.8,
        rationale="Most columns are referenced correctly.",
    )
    assert c.name == "grounding"
    assert c.label == "Grounding"
    assert c.score == 0.8
    assert "correctly" in c.rationale


# ---------------------------------------------------------------------------
# EvaluationResult - grade thresholds
# ---------------------------------------------------------------------------


def _make_result(score: float) -> EvaluationResult:
    return EvaluationResult(
        artifact_type="profiler",
        overall_score=score,
        grade=EvaluationResult.compute_grade(score),
        criteria=[],
        critique="",
        suggestions=[],
        judge_model="qwen3:14b",
        timestamp=make_timestamp(),
    )


def test_evaluation_result_grade_excellent():
    assert _make_result(0.90).grade == "excellent"
    assert _make_result(0.85).grade == "excellent"


def test_evaluation_result_grade_good():
    assert _make_result(0.84).grade == "good"
    assert _make_result(0.70).grade == "good"


def test_evaluation_result_grade_fair():
    assert _make_result(0.69).grade == "fair"
    assert _make_result(0.50).grade == "fair"


def test_evaluation_result_grade_poor():
    assert _make_result(0.49).grade == "poor"
    assert _make_result(0.0).grade == "poor"


# ---------------------------------------------------------------------------
# PipelineEvaluation - pipeline_score
# ---------------------------------------------------------------------------


def _make_eval(score: float, artifact_type: str = "profiler") -> EvaluationResult:
    return EvaluationResult(
        artifact_type=artifact_type,
        overall_score=score,
        grade=EvaluationResult.compute_grade(score),
        criteria=[],
        critique="",
        suggestions=[],
        judge_model="qwen3:14b",
        timestamp=make_timestamp(),
    )


def test_pipeline_evaluation_score_average():
    pe = PipelineEvaluation(
        profiler_eval=_make_eval(0.80),
        analyst_eval=AnalystEvaluationResult(
            artifact_type="analyst",
            overall_score=0.60,
            grade="fair",
            criteria=[],
            critique="",
            suggestions=[],
            judge_model="qwen3:14b",
            timestamp=make_timestamp(),
        ),
        reporter_eval=_make_eval(0.70, "reporter"),
    )
    assert abs(pe.pipeline_score - (0.80 + 0.60 + 0.70) / 3) < 1e-6


def test_pipeline_evaluation_partial():
    """pipeline_score ignores None artifact evaluations."""
    pe = PipelineEvaluation(
        profiler_eval=_make_eval(0.80),
        analyst_eval=None,
        reporter_eval=None,
    )
    assert abs(pe.pipeline_score - 0.80) < 1e-6


def test_pipeline_evaluation_all_none():
    pe = PipelineEvaluation()
    assert pe.pipeline_score == 0.0


# ---------------------------------------------------------------------------
# AnalystEvaluationResult - per_insight
# ---------------------------------------------------------------------------


def test_analyst_evaluation_result_per_insight():
    ins = InsightScore(
        title="High concentration in Ile-de-France",
        factual_correctness=0.9,
        relevance=0.8,
        actionability=0.7,
        note="Well-grounded in column data.",
    )
    result = AnalystEvaluationResult(
        artifact_type="analyst",
        overall_score=0.75,
        grade="good",
        criteria=[],
        critique="Solid insights.",
        suggestions=[],
        judge_model="qwen3:14b",
        timestamp=make_timestamp(),
        per_insight=[ins],
    )
    assert len(result.per_insight) == 1
    assert result.per_insight[0].title == "High concentration in Ile-de-France"
    assert result.per_insight[0].factual_correctness == 0.9


def test_insight_score_fields():
    ins = InsightScore(
        title="Test insight",
        factual_correctness=0.5,
        relevance=0.6,
        actionability=0.7,
        note="Some observation.",
    )
    assert ins.title == "Test insight"
    assert ins.factual_correctness == 0.5
    assert ins.relevance == 0.6
    assert ins.actionability == 0.7
    assert ins.note == "Some observation."


def test_analyst_result_default_per_insight_is_empty():
    result = AnalystEvaluationResult(
        artifact_type="analyst",
        overall_score=0.5,
        grade="fair",
        criteria=[],
        critique="",
        suggestions=[],
        judge_model="qwen3:14b",
        timestamp=make_timestamp(),
    )
    assert result.per_insight == []
