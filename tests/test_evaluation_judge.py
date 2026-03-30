"""Unit and integration tests for evaluation/llm_judge.py.

Unit tests mock call_llm_with_messages to avoid LLM calls.
Integration tests are marked @pytest.mark.integration and require Ollama.
"""

import json
import re

import pytest

from evaluation.llm_judge import (
    MAX_INSIGHT_FIELD_CHARS,
    MAX_SECTION_CHARS,
    build_judge_human_prompt,
    build_judge_system_prompt,
    parse_judge_response,
    truncate_for_judge,
)
from evaluation.rubrics import RUBRICS
from evaluation.schemas import AnalystEvaluationResult, EvaluationResult
from tools.profiler_engine import DataProfiler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profile(sample_df):
    return DataProfiler().profile(sample_df)


@pytest.fixture
def good_profiler_output():
    return """## Dataset Overview
This dataset contains 20 rows and 5 columns including age, salary, department.

## Data Quality Assessment
Missing values are present in the age column (10%) and score column (15%).
Overall quality rating: Fair - moderate missingness.

## Column-by-Column Analysis
- age: numeric, mean ~47.5, some missing values
- salary: numeric, range 30000-150000
- department: categorical, top value Engineering

## Statistical Highlights
The salary column has high dispersion with std > 30000.
The score column has 15% missing values.

## Key Takeaways & Recommendations
1. Impute missing age values using median.
2. Investigate high salary variance.
"""


@pytest.fixture
def good_insights():
    return [
        {
            "title": "High age variance",
            "observation": "Age ranges from 25 to 70 with significant gaps.",
            "hypothesis": "Diverse workforce or sampling bias.",
            "recommendation": "Segment analysis by age group.",
            "priority": "medium",
        },
        {
            "title": "Salary dispersion",
            "observation": "Salary std exceeds mean for some segments.",
            "hypothesis": "Mixed seniority levels in dataset.",
            "recommendation": "Stratify by department before modelling.",
            "priority": "high",
        },
    ]


@pytest.fixture
def valid_judge_json():
    """A well-formed judge JSON response for profiler artifact_type."""
    return json.dumps(
        {
            "grounding": {"score": 0.85, "rationale": "Column names are correctly referenced."},
            "completeness": {"score": 0.90, "rationale": "All required sections are present."},
            "clarity": {"score": 0.80, "rationale": "Language is accessible."},
            "specificity": {"score": 0.75, "rationale": "Uses real column names and numbers."},
            "overall_critique": "The profiler output is thorough and well-structured.",
            "suggestions": ["Add more detail on statistical highlights."],
        }
    )


@pytest.fixture
def valid_analyst_judge_json():
    return json.dumps(
        {
            "factual_correctness": {"score": 0.80, "rationale": "Observations match the data."},
            "relevance": {"score": 0.75, "rationale": "Insights are domain-relevant."},
            "actionability": {"score": 0.70, "rationale": "Recommendations are concrete."},
            "priority_calibration": {"score": 0.65, "rationale": "Priorities are reasonable."},
            "diversity": {"score": 0.80, "rationale": "Covers different aspects."},
            "per_insight": [
                {
                    "title": "High age variance",
                    "factual_correctness": 0.85,
                    "relevance": 0.80,
                    "actionability": 0.70,
                    "note": "Well-grounded in the age column statistics.",
                },
                {
                    "title": "Salary dispersion",
                    "factual_correctness": 0.75,
                    "relevance": 0.70,
                    "actionability": 0.75,
                    "note": "Good recommendation for stratified modelling.",
                },
            ],
            "overall_critique": "Insights are solid and cover key data patterns.",
            "suggestions": ["Consider adding a trend insight if date column is available."],
        }
    )


# ---------------------------------------------------------------------------
# truncate_for_judge
# ---------------------------------------------------------------------------


def test_truncate_for_judge_profiler(good_profiler_output):
    result = truncate_for_judge(good_profiler_output, "profiler")
    for kw in ["Overview", "Quality", "Column", "Statistical", "Takeaways"]:
        assert kw in result, f"Section '{kw}' missing after truncation"


def test_truncate_for_judge_section_body_length(good_profiler_output):
    result = truncate_for_judge(good_profiler_output, "profiler")
    parts = re.split(r"(?m)^#{1,3} .+$", result)
    for part in parts:
        stripped = part.strip().rstrip(".")
        assert len(stripped) <= MAX_SECTION_CHARS + 3, f"Body too long: {len(stripped)}"


def test_truncate_for_judge_analyst(good_insights):
    json_str = json.dumps(good_insights, ensure_ascii=False)
    result = truncate_for_judge(json_str, "analyst")
    parsed = json.loads(result)
    assert len(parsed) == len(good_insights)
    for ins in parsed:
        for field in ("title", "observation", "hypothesis", "recommendation"):
            val = ins.get(field, "")
            clean = val.rstrip(".")
            assert len(clean) <= MAX_INSIGHT_FIELD_CHARS, (
                f"Field '{field}' too long after truncation: {len(clean)}"
            )


def test_truncate_for_judge_no_crash_on_empty():
    assert truncate_for_judge("", "profiler") == ""
    assert truncate_for_judge("", "analyst") == ""
    assert truncate_for_judge("", "reporter") == ""


# ---------------------------------------------------------------------------
# parse_judge_response
# ---------------------------------------------------------------------------


def test_parse_judge_response_valid_json(valid_judge_json):
    result = parse_judge_response(valid_judge_json, "profiler")
    assert isinstance(result, EvaluationResult)
    assert result.artifact_type == "profiler"
    assert result.overall_score > 0.0
    assert len(result.criteria) == len(RUBRICS["profiler"])
    assert result.critique != ""


def test_parse_judge_response_with_markdown_fences(valid_judge_json):
    wrapped = f"```json\n{valid_judge_json}\n```"
    result = parse_judge_response(wrapped, "profiler")
    assert isinstance(result, EvaluationResult)
    assert result.overall_score > 0.0


def test_parse_judge_response_invalid_json():
    result = parse_judge_response("this is not json at all !!!", "profiler")
    assert result.overall_score == 0.0
    assert result.grade == "poor"
    assert "Evaluation failed" in result.critique


def test_parse_judge_response_analyst_per_insight(valid_analyst_judge_json):
    result = parse_judge_response(valid_analyst_judge_json, "analyst")
    assert isinstance(result, AnalystEvaluationResult)
    assert len(result.per_insight) == 2
    assert result.per_insight[0].title == "High age variance"
    assert result.per_insight[0].factual_correctness == 0.85


def test_parse_judge_response_weighted_score(valid_judge_json):
    result = parse_judge_response(valid_judge_json, "profiler")
    # 0.85*0.35 + 0.90*0.25 + 0.80*0.25 + 0.75*0.15
    expected = 0.85 * 0.35 + 0.90 * 0.25 + 0.80 * 0.25 + 0.75 * 0.15
    assert abs(result.overall_score - expected) < 1e-3


# ---------------------------------------------------------------------------
# build_judge_system_prompt
# ---------------------------------------------------------------------------


def test_build_judge_system_prompt_contains_criteria():
    rubric = RUBRICS["profiler"]
    prompt = build_judge_system_prompt(rubric)
    for entry in rubric:
        assert entry["name"] in prompt, f"Criterion '{entry['name']}' not in system prompt"


# ---------------------------------------------------------------------------
# build_judge_human_prompt
# ---------------------------------------------------------------------------


def test_build_judge_human_prompt_contains_artifact():
    rubric = RUBRICS["profiler"]
    artifact = "## Dataset Overview\nThis is a test artifact."
    prompt = build_judge_human_prompt(
        artifact=artifact,
        validation_report={"section_check": {"present": [], "missing": []}},
        rubric=rubric,
        artifact_type="profiler",
    )
    assert "Dataset Overview" in prompt


def test_build_judge_human_prompt_analyst_uses_analyst_template():
    rubric = RUBRICS["analyst"]
    artifact = json.dumps(
        [
            {
                "title": "test",
                "observation": "x",
                "hypothesis": "y",
                "recommendation": "z",
                "priority": "low",
            }
        ]
    )
    prompt = build_judge_human_prompt(
        artifact=artifact,
        validation_report={},
        rubric=rubric,
        artifact_type="analyst",
    )
    assert "per_insight" in prompt


# ---------------------------------------------------------------------------
# Integration tests (require running Ollama with EVAL_JUDGE_MODEL)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_evaluate_profiler_good_artifact(good_profiler_output, profile):
    from evaluation.llm_judge import EvaluationAgent

    agent = EvaluationAgent()
    result = agent.evaluate_profiler(good_profiler_output, profile)
    assert isinstance(result, EvaluationResult)
    assert result.overall_score >= 0.0
    assert result.artifact_type == "profiler"


@pytest.mark.integration
def test_evaluate_profiler_bad_artifact(profile):
    from evaluation.llm_judge import EvaluationAgent

    lorem = "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 10
    agent = EvaluationAgent()
    result = agent.evaluate_profiler(lorem, profile)
    assert isinstance(result, EvaluationResult)
    assert result.overall_score < 0.60


@pytest.mark.integration
def test_evaluate_analyst_returns_per_insight(good_insights, profile):
    from evaluation.llm_judge import EvaluationAgent

    agent = EvaluationAgent()
    result = agent.evaluate_analyst(good_insights, profile)
    assert isinstance(result, AnalystEvaluationResult)
    assert result.artifact_type == "analyst"


@pytest.mark.integration
def test_evaluate_reporter_good_artifact(good_profiler_output):
    from evaluation.llm_judge import EvaluationAgent

    agent = EvaluationAgent()
    reporter_output = (
        "## Executive Summary\nThis dataset contains 20 rows.\n"
        "## Dataset Description\nThe data covers employees.\n"
        "## Key Insights\nSalary shows high dispersion.\n"
        "## Visualizations\nNo charts generated.\n"
        "## Recommendations\n1. Impute missing values.\n"
        "## Limitations & Next Steps\nCausality is not established.\n"
    )
    result = agent.evaluate_reporter(reporter_output, good_profiler_output)
    assert isinstance(result, EvaluationResult)
    assert result.artifact_type == "reporter"
