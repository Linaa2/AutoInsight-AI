"""Tests for the LLM-as-Judge evaluation subsystem (P10).

Unit tests run without a live LLM — EvaluationAgent methods are exercised
via ``unittest.mock.patch`` on ``call_llm_with_messages``.

Integration tests (``@pytest.mark.integration``) require a live LLM and are
excluded from the default ``make test`` run.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from evaluation.config import (
    EVAL_EXCELLENT_THRESHOLD,
    EVAL_FAIR_THRESHOLD,
)
from evaluation.llm_judge import (
    EvaluationAgent,
    parse_judge_response,
    truncate_for_judge,
)
from evaluation.rubrics import REQUIRED_SECTIONS, RUBRICS
from evaluation.schemas import (
    AnalystEvaluationResult,
    EvaluationCriterion,
    EvaluationResult,
    PipelineEvaluation,
)
from evaluation.validators import (
    check_column_references,
    check_insight_fields,
    check_required_sections,
    check_statistic_accuracy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PROFILE_DATA: dict = {
    "shape": [1000, 5],
    "columns": {
        "age": {"dtype": "float64", "dtype_category": "numeric", "missing_pct": 5.0},
        "salary": {"dtype": "float64", "dtype_category": "numeric", "missing_pct": 0.0},
        "region": {"dtype": "object", "dtype_category": "categorical", "missing_pct": 2.0},
        "date": {"dtype": "datetime64", "dtype_category": "datetime", "missing_pct": 0.0},
        "active": {"dtype": "bool", "dtype_category": "boolean", "missing_pct": 0.0},
    },
    "missing": {"age": 5.0},
    "stats": {
        "duplicates_count": 10,
        "duplicates_pct": 1.0,
        "total_missing_count": 50,
        "total_missing_pct": 5.0,
        "memory_mb": 0.5,
        "numeric_cols": 2,
        "categorical_cols": 1,
        "datetime_cols": 1,
        "boolean_cols": 1,
    },
}

SAMPLE_INSIGHTS: list[dict] = [
    {
        "title": "Age distribution skewed",
        "observation": "The age column shows high right-skew.",
        "hypothesis": "Sampling bias towards younger users.",
        "recommendation": "Apply log transform before modelling.",
        "priority": "high",
    },
    {
        "title": "Salary outliers present",
        "observation": "Salary std exceeds the mean.",
        "hypothesis": "Data entry errors or executive outliers.",
        "recommendation": "Cap values at 99th percentile.",
        "priority": "medium",
    },
]

PROFILER_SECTION_TEXT = """
## Dataset Overview
1000 rows, 5 columns.

## Data Quality Assessment
5.0% missing values. 1.0% duplicates.

## Column-by-Column Analysis
Details about age, salary, region, date, active.

## Statistical Highlights
Skew detected in age column.

## Key Takeaways & Recommendations
Apply log transform.
"""


def _make_profiler_response_json() -> str:
    return json.dumps(
        {
            "criteria": {
                "grounding": {"score": 0.82, "rationale": "Columns correctly cited."},
                "completeness": {"score": 0.90, "rationale": "All sections present."},
                "clarity": {"score": 0.75, "rationale": "Clear writing."},
                "specificity": {"score": 0.70, "rationale": "Good numbers cited."},
            },
            "critique": "The report is comprehensive and accurate.",
            "suggestions": ["Add correlation hints.", "Expand datetime analysis."],
        }
    )


def _make_analyst_response_json() -> str:
    return json.dumps(
        {
            "criteria": {
                "factual_correctness": {"score": 0.85, "rationale": "Claims match profile."},
                "relevance": {"score": 0.80, "rationale": "Insights are business-relevant."},
                "actionability": {"score": 0.75, "rationale": "Recs are implementable."},
                "priority_calibration": {
                    "score": 0.70,
                    "rationale": "Priority assignments reasonable.",
                },
                "diversity": {"score": 0.90, "rationale": "Insights cover different aspects."},
            },
            "critique": "Insights are diverse and actionable.",
            "suggestions": ["Add temporal analysis."],
            "per_insight": [
                {
                    "title": "Age distribution skewed",
                    "factual_correctness": 0.85,
                    "relevance": 0.80,
                    "actionability": 0.75,
                    "note": "Well-grounded in skewness statistics.",
                },
                {
                    "title": "Salary outliers present",
                    "factual_correctness": 0.80,
                    "relevance": 0.70,
                    "actionability": 0.90,
                    "note": "Concrete recommendation.",
                },
            ],
        }
    )


def _make_reporter_response_json() -> str:
    return json.dumps(
        {
            "criteria": {
                "faithfulness": {"score": 0.88, "rationale": "Key findings represented."},
                "coherence": {"score": 0.82, "rationale": "Narrative flows well."},
                "language": {"score": 0.90, "rationale": "Accessible writing."},
                "completeness": {"score": 0.85, "rationale": "All sections present."},
                "actionability": {"score": 0.75, "rationale": "Concrete recommendations."},
            },
            "critique": "Report faithfully synthesises all findings.",
            "suggestions": ["Mention data caveats in Limitations."],
        }
    )


# ---------------------------------------------------------------------------
# truncate_for_judge
# ---------------------------------------------------------------------------


def test_truncate_preserves_all_sections() -> None:
    text = "## Section A\nBody A.\n## Section B\nBody B.\n## Section C\nBody C."
    result = truncate_for_judge(text, max_chars_per_section=500)
    assert "## Section A" in result
    assert "## Section B" in result
    assert "## Section C" in result


def test_truncate_caps_body_per_section() -> None:
    long_body = "x" * 1000
    text = f"## Heading\n{long_body}"
    result = truncate_for_judge(text, max_chars_per_section=50)
    # Find the body in the result (after the heading line)
    body_part = result.split("## Heading\n", 1)[-1]
    # Body should be capped + truncation marker
    assert len(body_part) <= 60 + len("\n[...truncated...]")
    assert "[...truncated...]" in result


def test_truncate_short_text_unchanged() -> None:
    text = "Short text without headings."
    assert truncate_for_judge(text, max_chars_per_section=500) == text


# ---------------------------------------------------------------------------
# check_required_sections
# ---------------------------------------------------------------------------


def test_required_sections_all_present() -> None:
    result = check_required_sections(PROFILER_SECTION_TEXT, "profiler")
    assert result["missing"] == [], f"Unexpected missing: {result['missing']}"
    assert len(result["present"]) == len(REQUIRED_SECTIONS["profiler"])


def test_required_sections_missing_one() -> None:
    text = PROFILER_SECTION_TEXT.replace("Statistical Highlights", "REMOVED")
    result = check_required_sections(text, "profiler")
    assert "Statistical Highlights" in result["missing"]


# ---------------------------------------------------------------------------
# check_column_references
# ---------------------------------------------------------------------------


def test_column_references_present() -> None:
    text = "The age column has 5% missingness. The salary distribution is wide."
    result = check_column_references(text, SAMPLE_PROFILE_DATA)
    assert "age" in result["present"]
    assert "salary" in result["present"]


def test_column_references_unmentioned() -> None:
    text = "Only age is discussed here."
    result = check_column_references(text, SAMPLE_PROFILE_DATA)
    # salary, region, date, active should all be unmentioned
    assert "salary" in result["unmentioned"]
    assert "region" in result["unmentioned"]


# ---------------------------------------------------------------------------
# check_statistic_accuracy
# ---------------------------------------------------------------------------


def test_stat_accuracy_verified() -> None:
    # Text explicitly contains the values from SAMPLE_PROFILE_DATA
    text = "Dataset has 1000 rows, 5 columns. Total missing: 5.0%. Duplicates: 1.0%."
    result = check_statistic_accuracy(text, SAMPLE_PROFILE_DATA)
    assert "row_count" in result["verified"]
    assert "total_missing_pct" in result["verified"]


def test_stat_accuracy_unverified() -> None:
    # Text has no matching numbers
    text = "This text mentions nothing useful about the data."
    result = check_statistic_accuracy(text, SAMPLE_PROFILE_DATA)
    assert "row_count" in result["unverified"]
    assert result["ground_truth"]  # ground_truth always populated


# ---------------------------------------------------------------------------
# check_insight_fields
# ---------------------------------------------------------------------------


def test_insight_fields_valid() -> None:
    result = check_insight_fields(SAMPLE_INSIGHTS)
    assert result["valid_count"] == 2
    assert result["issues"] == []


def test_insight_fields_missing_title() -> None:
    bad_insight = {k: v for k, v in SAMPLE_INSIGHTS[0].items() if k != "title"}
    result = check_insight_fields([bad_insight])
    assert result["valid_count"] == 0
    assert len(result["issues"]) == 1
    assert "title" in result["issues"][0]


# ---------------------------------------------------------------------------
# parse_judge_response
# ---------------------------------------------------------------------------


def test_parse_profiler_response_valid() -> None:
    raw = _make_profiler_response_json()
    result = parse_judge_response(raw, "profiler")
    assert isinstance(result, EvaluationResult)
    assert result.artifact_type == "profiler"
    assert 0.0 <= result.overall_score <= 1.0
    assert result.grade in ("excellent", "good", "fair", "poor")
    assert len(result.criteria) == len(RUBRICS["profiler"])


def test_parse_analyst_response_with_per_insight() -> None:
    raw = _make_analyst_response_json()
    result = parse_judge_response(raw, "analyst")
    assert isinstance(result, AnalystEvaluationResult)
    assert len(result.per_insight) == 2
    assert result.per_insight[0].title == "Age distribution skewed"
    assert 0.0 <= result.per_insight[0].factual_correctness <= 1.0


def test_parse_invalid_json_returns_fallback() -> None:
    result = parse_judge_response("this is not json at all !!!!", "profiler")
    assert isinstance(result, EvaluationResult)
    assert result.overall_score == 0.5
    assert result.grade == "fair"


def test_parse_partial_criteria_fill_defaults() -> None:
    # Only 'grounding' is present; others should default to 0.5
    partial = json.dumps(
        {
            "criteria": {
                "grounding": {"score": 1.0, "rationale": "Perfect grounding."},
            },
            "critique": "Partial response.",
            "suggestions": [],
        }
    )
    result = parse_judge_response(partial, "profiler")
    assert isinstance(result, EvaluationResult)
    missing_criteria = [c for c in result.criteria if c.name != "grounding"]
    for c in missing_criteria:
        assert c.score == 0.5, f"Expected 0.5 for {c.name}, got {c.score}"


# ---------------------------------------------------------------------------
# Weighted score and grade
# ---------------------------------------------------------------------------


def test_overall_score_is_weighted_average() -> None:
    # profiler weights: grounding=0.35, completeness=0.25, clarity=0.25, specificity=0.15
    criteria = [
        EvaluationCriterion("grounding", "Grounding", 0.8, "ok", 0.35),
        EvaluationCriterion("completeness", "Completeness", 0.6, "ok", 0.25),
        EvaluationCriterion("clarity", "Clarity", 0.7, "ok", 0.25),
        EvaluationCriterion("specificity", "Specificity", 0.5, "ok", 0.15),
    ]
    expected = 0.8 * 0.35 + 0.6 * 0.25 + 0.7 * 0.25 + 0.5 * 0.15
    result = EvaluationResult(
        artifact_type="profiler",
        overall_score=sum(c.score * c.weight for c in criteria),
        grade="good",
        criteria=criteria,
        critique="",
        suggestions=[],
        judge_model="test",
        timestamp="",
    )
    assert abs(result.overall_score - expected) < 1e-9


def test_compute_grade_excellent() -> None:
    assert EvaluationResult.compute_grade(EVAL_EXCELLENT_THRESHOLD) == "excellent"
    assert EvaluationResult.compute_grade(1.0) == "excellent"


def test_compute_grade_poor() -> None:
    assert EvaluationResult.compute_grade(0.0) == "poor"
    assert EvaluationResult.compute_grade(EVAL_FAIR_THRESHOLD - 0.01) == "poor"


# ---------------------------------------------------------------------------
# EvaluationAgent — LLM call counts
# ---------------------------------------------------------------------------

_LLM_PATH = "evaluation.llm_judge.call_llm_with_messages"


def test_evaluate_profiler_calls_llm_once() -> None:
    with patch(_LLM_PATH, return_value=_make_profiler_response_json()) as mock_llm:
        agent = EvaluationAgent()
        result = agent.evaluate_profiler(PROFILER_SECTION_TEXT, SAMPLE_PROFILE_DATA)
    mock_llm.assert_called_once()
    assert isinstance(result, EvaluationResult)


def test_evaluate_analyst_calls_llm_once() -> None:
    with patch(_LLM_PATH, return_value=_make_analyst_response_json()) as mock_llm:
        agent = EvaluationAgent()
        result = agent.evaluate_analyst(SAMPLE_INSIGHTS, PROFILER_SECTION_TEXT, SAMPLE_PROFILE_DATA)
    mock_llm.assert_called_once()
    assert isinstance(result, AnalystEvaluationResult)


def test_evaluate_reporter_calls_llm_once() -> None:
    report_text = (
        "## Executive Summary\nGood dataset.\n"
        "## Dataset Description\nDetails.\n"
        "## Key Insights\nInsight A.\n"
        "## Visualizations\nChart shown.\n"
        "## Recommendations\nDo X.\n"
        "## Limitations & Next Steps\nCaveats here."
    )
    with patch(_LLM_PATH, return_value=_make_reporter_response_json()) as mock_llm:
        agent = EvaluationAgent()
        result = agent.evaluate_reporter(report_text, PROFILER_SECTION_TEXT, "insights md")
    mock_llm.assert_called_once()
    assert isinstance(result, EvaluationResult)


# ---------------------------------------------------------------------------
# Context injection (critic / uncertainty)
# ---------------------------------------------------------------------------


def test_evaluate_analyst_with_critic_context() -> None:
    critiques = [
        {
            "insight_title": "Age distribution skewed",
            "verdict": "partially_supported",
            "weaknesses": "Population density not considered.",
        }
    ]
    with patch(_LLM_PATH, return_value=_make_analyst_response_json()) as mock_llm:
        agent = EvaluationAgent()
        agent.evaluate_analyst(
            SAMPLE_INSIGHTS,
            PROFILER_SECTION_TEXT,
            SAMPLE_PROFILE_DATA,
            critiques=critiques,
        )
    human_prompt: str = mock_llm.call_args.kwargs["human"]
    assert "partially_supported" in human_prompt
    assert "Age distribution skewed" in human_prompt


def test_evaluate_analyst_with_uncertainty_context() -> None:
    confidence_scores = [
        {
            "insight_title": "Salary outliers present",
            "confidence_score": 41,
            "confidence_level": "low",
            "summary": "38% missing in salary column.",
        }
    ]
    with patch(_LLM_PATH, return_value=_make_analyst_response_json()) as mock_llm:
        agent = EvaluationAgent()
        agent.evaluate_analyst(
            SAMPLE_INSIGHTS,
            PROFILER_SECTION_TEXT,
            SAMPLE_PROFILE_DATA,
            confidence_scores=confidence_scores,
        )
    human_prompt: str = mock_llm.call_args.kwargs["human"]
    assert "low confidence" in human_prompt
    assert "Salary outliers present" in human_prompt


# ---------------------------------------------------------------------------
# PipelineEvaluation
# ---------------------------------------------------------------------------


def _make_eval_result(artifact_type: str, score: float) -> EvaluationResult:
    rubric = RUBRICS[artifact_type]
    criteria = [
        EvaluationCriterion(str(c["name"]), str(c["label"]), score, "ok", float(c["weight"]))
        for c in rubric
    ]
    return EvaluationResult(
        artifact_type=artifact_type,
        overall_score=score,
        grade=EvaluationResult.compute_grade(score),
        criteria=criteria,
        critique="",
        suggestions=[],
        judge_model="test",
        timestamp="",
    )


def test_pipeline_evaluation_score_average() -> None:
    pe = PipelineEvaluation(
        profiler_eval=_make_eval_result("profiler", 0.8),
        reporter_eval=_make_eval_result("reporter", 0.6),
    )
    assert abs(pe.pipeline_score - 0.7) < 1e-9


def test_pipeline_evaluation_grade_derived() -> None:
    pe = PipelineEvaluation(
        profiler_eval=_make_eval_result("profiler", 0.9),
        analyst_eval=None,
        reporter_eval=_make_eval_result("reporter", 0.9),
    )
    assert pe.pipeline_grade == "excellent"


# ---------------------------------------------------------------------------
# Fallback on LLM failure
# ---------------------------------------------------------------------------


def test_evaluate_profiler_no_llm_fallback() -> None:
    with patch(_LLM_PATH, side_effect=RuntimeError("LLM unavailable")):
        agent = EvaluationAgent()
        result = agent.evaluate_profiler(PROFILER_SECTION_TEXT, SAMPLE_PROFILE_DATA)
    assert isinstance(result, EvaluationResult)
    assert result.overall_score == 0.5
    assert result.grade == "fair"
    assert "LLM unavailable" in result.critique or "unavailable" in result.critique.lower()


# ---------------------------------------------------------------------------
# Rubric weight integrity
# ---------------------------------------------------------------------------


def test_rubric_weights_sum_to_one() -> None:
    for artifact_type, criteria in RUBRICS.items():
        total = sum(float(c["weight"]) for c in criteria)
        assert abs(total - 1.0) < 1e-9, f"{artifact_type} weights sum to {total}"


# ---------------------------------------------------------------------------
# Integration tests (require a live LLM)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_evaluate_profiler_live() -> None:
    agent = EvaluationAgent()
    result = agent.evaluate_profiler(PROFILER_SECTION_TEXT, SAMPLE_PROFILE_DATA)
    assert isinstance(result, EvaluationResult)
    assert len(result.criteria) == len(RUBRICS["profiler"])
    assert 0.0 <= result.overall_score <= 1.0
    assert result.grade in ("excellent", "good", "fair", "poor")


@pytest.mark.integration
def test_evaluate_analyst_live() -> None:
    agent = EvaluationAgent()
    result = agent.evaluate_analyst(SAMPLE_INSIGHTS, PROFILER_SECTION_TEXT, SAMPLE_PROFILE_DATA)
    assert isinstance(result, AnalystEvaluationResult)
    assert 0.0 <= result.overall_score <= 1.0


@pytest.mark.integration
def test_evaluate_reporter_live() -> None:
    report_text = (
        "## Executive Summary\nGood dataset.\n"
        "## Dataset Description\nDetails.\n"
        "## Key Insights\nInsight A.\n"
        "## Visualizations\nChart shown.\n"
        "## Recommendations\nDo X.\n"
        "## Limitations & Next Steps\nCaveats here."
    )
    agent = EvaluationAgent()
    result = agent.evaluate_reporter(report_text, PROFILER_SECTION_TEXT, "insights md")
    assert isinstance(result, EvaluationResult)
    assert result.suggestions  # At least one suggestion expected
