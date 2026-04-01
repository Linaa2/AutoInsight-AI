"""Tests for the UncertaintyEstimator agent (Phase 8).

All tests are unit tests — no live LLM required. The LLM-based driver
(_compute_llm_scores) is patched with controlled mock responses.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from agents.uncertainty import (
    UncertaintyEstimator,
    UncertaintyFormatter,
    _compute_data_quality,
    _compute_specificity,
    _determine_level,
    _extract_json,
    uncertainty_node,
)
from evaluation.config import EVAL_UNCERTAINTY_HIGH_THRESHOLD, EVAL_UNCERTAINTY_MEDIUM_THRESHOLD

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def large_clean_profile() -> dict[str, Any]:
    """Profile for a large, clean dataset (should yield high data quality score)."""
    return {
        "shape": [1500, 8],
        "columns": {
            "order_id": {"dtype_category": "numeric", "missing_pct": 0.0},
            "region": {"dtype_category": "categorical", "missing_pct": 0.5},
            "revenue": {
                "dtype_category": "numeric",
                "missing_pct": 0.8,
                "mean": 45234.0,
                "std": 12000.0,
            },
        },
        "stats": {
            "total_missing_pct": 0.8,
            "duplicates_pct": 0.2,
        },
    }


@pytest.fixture
def small_dirty_profile() -> dict[str, Any]:
    """Profile for a small, dirty dataset (should yield low data quality score)."""
    return {
        "shape": [80, 4],
        "columns": {
            "age": {"dtype_category": "numeric", "missing_pct": 45.0},
        },
        "stats": {
            "total_missing_pct": 45.0,
            "duplicates_pct": 15.0,
        },
    }


@pytest.fixture
def medium_profile() -> dict[str, Any]:
    """Profile for a medium-sized dataset with moderate quality."""
    return {
        "shape": [300, 5],
        "columns": {
            "salary": {
                "dtype_category": "numeric",
                "missing_pct": 12.0,
                "mean": 50000.0,
                "std": 15000.0,
            },
            "department": {"dtype_category": "categorical", "missing_pct": 5.0},
        },
        "stats": {
            "total_missing_pct": 12.0,
            "duplicates_pct": 5.0,
        },
    }


@pytest.fixture
def specific_insight() -> dict[str, Any]:
    """Insight with concrete column references and numbers."""
    return {
        "title": "Revenue concentration in one region",
        "observation": "32% of orders (480/1500) come from the revenue and region columns",
        "hypothesis": "Logistics and population density explain the concentration",
        "recommendation": "Normalize sales by regional population before deciding on expansion strategy",
        "priority": "high",
    }


@pytest.fixture
def vague_insight() -> dict[str, Any]:
    """Insight with no concrete details."""
    return {
        "title": "Some missing values",
        "observation": "There are some missing values in the dataset",
        "hypothesis": "Data collection issues",
        "recommendation": "Investigate further",
        "priority": "low",
    }


@pytest.fixture
def mock_critique() -> dict[str, Any]:
    """A mock CriticAgent output following the critique contract."""
    return {
        "insight_title": "Revenue concentration in one region",
        "strengths": "Backed by exact counts: 480/1500 orders.",
        "weaknesses": "Population density not accounted for — the concentration may be less extreme.",
        "alternatives": "Marketing spend or warehouse location could be the real driver.",
        "confidence": "medium",
        "verdict": "partially_supported",
    }


@pytest.fixture
def sample_insights(specific_insight: dict[str, Any]) -> list[dict[str, Any]]:
    """List with two insights for estimate_all tests."""
    return [
        specific_insight,
        {
            "title": "High salary dispersion",
            "observation": "Salary std=15000 exceeds 30% of the salary mean (50000)",
            "hypothesis": "Different seniority levels or departments",
            "recommendation": "Segment employees by department before comparing compensation",
            "priority": "medium",
        },
    ]


@pytest.fixture
def mock_llm_scores() -> dict[str, Any]:
    """Controlled LLM response for patching _compute_llm_scores."""
    return {
        "statistical_evidence": {"score": 18, "reason": "Clear percentage, large sample"},
        "critic_assessment": {"score": 14, "reason": "Critic concern is valid but non-fatal"},
    }


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_valid() -> None:
    raw = '{"statistical_evidence": {"score": 18, "reason": "ok"}}'
    result = _extract_json(raw)
    assert result is not None
    assert result["statistical_evidence"]["score"] == 18


def test_extract_json_with_noise() -> None:
    raw = 'Some text before\n```json\n{"a": 1}\n```\nSome text after'
    result = _extract_json(raw)
    assert result == {"a": 1}


def test_extract_json_invalid_returns_none() -> None:
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None


# ---------------------------------------------------------------------------
# _compute_data_quality (rule-based)
# ---------------------------------------------------------------------------


def test_data_quality_large_clean(large_clean_profile: dict[str, Any]) -> None:
    score, reason = _compute_data_quality(large_clean_profile)
    # 1500 rows -> 10 + <5% missing -> 10 + <1% duplicates -> 5 = 25
    assert score == 25
    assert "1500" in reason


def test_data_quality_small_dirty(small_dirty_profile: dict[str, Any]) -> None:
    score, reason = _compute_data_quality(small_dirty_profile)
    # 80 rows -> 1 + >40% missing -> 0 + >10% dup -> 0 = 1
    assert score == 1
    assert "80" in reason


def test_data_quality_medium(medium_profile: dict[str, Any]) -> None:
    score, _reason = _compute_data_quality(medium_profile)
    # 300 rows -> 7 + 5-20% missing -> 6 + 1-10% dup -> 3 = 16
    assert score == 16


def test_data_quality_high_duplicates() -> None:
    profile: dict[str, Any] = {
        "shape": [500, 3],
        "columns": {},
        "stats": {"total_missing_pct": 2.0, "duplicates_pct": 20.0},
    }
    score, _ = _compute_data_quality(profile)
    # 500 rows -> 7 + <5% missing -> 10 + >10% dup -> 0 = 17
    assert score == 17


def test_data_quality_missing_stats_defaults_to_zero() -> None:
    profile: dict[str, Any] = {"shape": [200, 2], "columns": {}, "stats": {}}
    score, _ = _compute_data_quality(profile)
    # 200 rows -> 4 + 0% missing -> 10 + 0% dup -> 5 = 19
    assert score == 19


# ---------------------------------------------------------------------------
# _compute_specificity (rule-based)
# ---------------------------------------------------------------------------


def test_specificity_column_and_numbers(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
) -> None:
    score, reason = _compute_specificity(specific_insight, large_clean_profile)
    # "revenue" and "region" appear in text -> 12 pts
    # "32", "480", "1500" -> >=2 numbers -> 11 pts
    # Recommendation >=8 words and not weak -> 2 pts
    # Total = 25
    assert score == 25
    assert "revenue" in reason or "region" in reason


def test_specificity_vague(
    vague_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
) -> None:
    score, reason = _compute_specificity(vague_insight, large_clean_profile)
    # No column names in text, no numbers -> 0 pts, no actionable rec -> 0
    assert score == 0
    assert "no concrete" in reason


def test_specificity_one_column_one_number(large_clean_profile: dict[str, Any]) -> None:
    insight: dict[str, Any] = {
        "title": "Revenue outlier",
        "observation": "The revenue column has a single outlier at value 1000000",
        "hypothesis": "Data entry error",
        "recommendation": "Remove the outlier before modelling and validate with the data owner",
        "priority": "high",
    }
    score, _ = _compute_specificity(insight, large_clean_profile)
    # "revenue" -> 8 pts; one number "1000000" -> 8 pts; rec is actionable -> 2 pts
    assert score == 18


def test_specificity_weak_recommendation(large_clean_profile: dict[str, Any]) -> None:
    insight: dict[str, Any] = {
        "title": "Issue",
        "observation": "The revenue column shows 45% missing values and region has anomalies",
        "hypothesis": "Data pipeline issue",
        "recommendation": "Investigate further",
        "priority": "medium",
    }
    score, _ = _compute_specificity(insight, large_clean_profile)
    # 2 cols -> 12, >=2 numbers (45) 1 actually... "45" -> 8 pts; weak rec -> 0
    # "revenue", "region" -> 12, "45" 1 number -> 8, weak -> 0 = 20
    assert score == 20


# ---------------------------------------------------------------------------
# _determine_level
# ---------------------------------------------------------------------------


def test_determine_level_high() -> None:
    assert _determine_level(EVAL_UNCERTAINTY_HIGH_THRESHOLD) == "high"
    assert _determine_level(100) == "high"
    assert _determine_level(85) == "high"


def test_determine_level_medium() -> None:
    assert _determine_level(EVAL_UNCERTAINTY_MEDIUM_THRESHOLD) == "medium"
    assert _determine_level(65) == "medium"
    assert _determine_level(EVAL_UNCERTAINTY_HIGH_THRESHOLD - 1) == "medium"


def test_determine_level_low() -> None:
    assert _determine_level(0) == "low"
    assert _determine_level(30) == "low"
    assert _determine_level(EVAL_UNCERTAINTY_MEDIUM_THRESHOLD - 1) == "low"


# ---------------------------------------------------------------------------
# UncertaintyEstimator.estimate — fallback behaviors
# ---------------------------------------------------------------------------


def test_fallback_no_critique(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
) -> None:
    """When no critique is provided, critic_assessment should default to 12."""
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        # LLM fails
        mock_llm.side_effect = RuntimeError("LLM unavailable")
        result = estimator.estimate(specific_insight, {}, large_clean_profile)

    drivers = result["drivers"]
    assert drivers["critic_assessment"]["score"] == 12
    assert "No critique" in drivers["critic_assessment"]["reason"]


def test_fallback_verdict_supported(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
) -> None:
    """When LLM fails but verdict='supported', critic score should be 20."""
    critique = {
        "insight_title": specific_insight["title"],
        "verdict": "supported",
        "confidence": "high",
        "strengths": "Solid data",
        "weaknesses": "",
        "alternatives": "",
    }
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.side_effect = RuntimeError("LLM unavailable")
        result = estimator.estimate(specific_insight, critique, large_clean_profile)

    assert result["drivers"]["critic_assessment"]["score"] == 20


def test_fallback_verdict_weak(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
) -> None:
    """When LLM fails but verdict='weak', critic score should be 5."""
    critique = {
        "insight_title": specific_insight["title"],
        "verdict": "weak",
        "confidence": "low",
        "strengths": "",
        "weaknesses": "Fundamental flaw",
        "alternatives": "Alternative is stronger",
    }
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.side_effect = RuntimeError("LLM unavailable")
        result = estimator.estimate(specific_insight, critique, large_clean_profile)

    assert result["drivers"]["critic_assessment"]["score"] == 5


def test_llm_scores_parsed_correctly(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
    mock_critique: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    """Valid LLM JSON is parsed into driver scores correctly."""
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = estimator.estimate(specific_insight, mock_critique, large_clean_profile)

    drivers = result["drivers"]
    assert drivers["statistical_evidence"]["score"] == 18
    assert drivers["critic_assessment"]["score"] == 14


def test_llm_scores_invalid_json_falls_back(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
    mock_critique: dict[str, Any],
) -> None:
    """Garbage LLM response falls back to defaults without crashing."""
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = "This is not JSON at all"
        result = estimator.estimate(specific_insight, mock_critique, large_clean_profile)

    drivers = result["drivers"]
    assert drivers["statistical_evidence"]["score"] == 12  # default
    # critic_assessment uses verdict fallback since critique is present
    assert drivers["critic_assessment"]["score"] == 12  # partially_supported default


# ---------------------------------------------------------------------------
# UncertaintyEstimator.estimate — structure checks
# ---------------------------------------------------------------------------


def test_estimate_returns_all_required_fields(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
    mock_critique: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = estimator.estimate(specific_insight, mock_critique, large_clean_profile)

    assert "insight_title" in result
    assert "confidence_score" in result
    assert "confidence_level" in result
    assert "drivers" in result
    assert "summary" in result

    drivers = result["drivers"]
    for key in ("data_quality", "specificity", "statistical_evidence", "critic_assessment"):
        assert key in drivers
        assert "score" in drivers[key]
        assert "max" in drivers[key]
        assert "reason" in drivers[key]
        assert drivers[key]["max"] == 25


def test_estimate_confidence_score_equals_driver_sum(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
    mock_critique: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = estimator.estimate(specific_insight, mock_critique, large_clean_profile)

    drivers = result["drivers"]
    expected_total = sum(d["score"] for d in drivers.values())
    assert result["confidence_score"] == expected_total


def test_estimate_title_matches_insight(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = estimator.estimate(specific_insight, {}, large_clean_profile)

    assert result["insight_title"] == specific_insight["title"]


# ---------------------------------------------------------------------------
# UncertaintyEstimator.estimate_all
# ---------------------------------------------------------------------------


def test_estimate_all_returns_one_per_insight(
    sample_insights: list[dict[str, Any]],
    large_clean_profile: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = estimator.estimate_all(sample_insights, [], large_clean_profile)

    assert len(result["confidence_scores"]) == len(sample_insights)


def test_estimate_all_batches_llm_call_once_for_typical_insight_count(
    sample_insights: list[dict[str, Any]],
    large_clean_profile: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    estimator = UncertaintyEstimator()
    batched = {
        "results": [
            {
                "index": i,
                "statistical_evidence": mock_llm_scores["statistical_evidence"],
                "critic_assessment": mock_llm_scores["critic_assessment"],
            }
            for i in range(len(sample_insights))
        ]
    }
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(batched)
        result = estimator.estimate_all(sample_insights, [], large_clean_profile)

    mock_llm.assert_called_once()
    assert len(result["confidence_scores"]) == len(sample_insights)


def test_estimate_all_empty_insights(large_clean_profile: dict[str, Any]) -> None:
    estimator = UncertaintyEstimator()
    result = estimator.estimate_all([], [], large_clean_profile)
    assert result["confidence_scores"] == []
    assert result["uncertainty_output"] == ""


def test_estimate_all_maps_critiques_by_title(
    sample_insights: list[dict[str, Any]],
    large_clean_profile: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    """Critiques are matched to insights by title; unmatched insights get empty critique."""
    critiques = [
        {
            "insight_title": sample_insights[0]["title"],
            "verdict": "supported",
            "confidence": "high",
            "strengths": "Good data",
            "weaknesses": "",
            "alternatives": "",
        }
    ]
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = estimator.estimate_all(sample_insights, critiques, large_clean_profile)

    # Both insights scored without crash
    assert len(result["confidence_scores"]) == 2


def test_estimate_all_returns_uncertainty_output(
    sample_insights: list[dict[str, Any]],
    large_clean_profile: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = estimator.estimate_all(sample_insights, [], large_clean_profile)

    assert isinstance(result["uncertainty_output"], str)
    assert len(result["uncertainty_output"]) > 0


def test_estimate_all_batch_unparseable_response_falls_back_per_insight(
    sample_insights: list[dict[str, Any]],
    large_clean_profile: dict[str, Any],
) -> None:
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = "not valid json"
        result = estimator.estimate_all(sample_insights, [], large_clean_profile)

    assert len(result["confidence_scores"]) == len(sample_insights)
    for score in result["confidence_scores"]:
        assert score["drivers"]["statistical_evidence"]["score"] == 12
        assert score["drivers"]["critic_assessment"]["score"] == 12


# ---------------------------------------------------------------------------
# UncertaintyFormatter
# ---------------------------------------------------------------------------


def test_to_markdown_contains_all_titles(
    sample_insights: list[dict[str, Any]],
    large_clean_profile: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    estimator = UncertaintyEstimator()
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = estimator.estimate_all(sample_insights, [], large_clean_profile)

    md = result["uncertainty_output"]
    for insight in sample_insights:
        assert insight["title"] in md


def test_to_markdown_empty_scores() -> None:
    assert UncertaintyFormatter.to_markdown([]) == ""


def test_to_markdown_contains_score_columns() -> None:
    scores = [
        {
            "insight_title": "Test insight",
            "confidence_score": 76,
            "confidence_level": "medium",
            "drivers": {
                "data_quality": {"score": 22, "max": 25, "reason": "Good"},
                "specificity": {"score": 20, "max": 25, "reason": "OK"},
                "statistical_evidence": {"score": 18, "max": 25, "reason": "Reasonable"},
                "critic_assessment": {"score": 16, "max": 25, "reason": "Minor concern"},
            },
            "summary": "Medium confidence.",
        }
    ]
    md = UncertaintyFormatter.to_markdown(scores)
    assert "76%" in md
    assert "medium" in md
    assert "22/25" in md


# ---------------------------------------------------------------------------
# uncertainty_node (standalone LangGraph node)
# ---------------------------------------------------------------------------


def test_uncertainty_node_writes_correct_keys(
    sample_insights: list[dict[str, Any]],
    large_clean_profile: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    state: dict[str, Any] = {
        "insights": sample_insights,
        "critiques": [],
        "profile_data": large_clean_profile,
    }
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = uncertainty_node(state)

    assert "confidence_scores" in result
    assert "uncertainty_output" in result
    assert len(result["confidence_scores"]) == len(sample_insights)


def test_uncertainty_node_empty_insights() -> None:
    state: dict[str, Any] = {"insights": [], "critiques": [], "profile_data": {}}
    result = uncertainty_node(state)
    assert result["confidence_scores"] == []
    assert result["uncertainty_output"] == ""


def test_uncertainty_node_no_insights_key() -> None:
    """Node should not crash when 'insights' is absent from state."""
    result = uncertainty_node({})
    assert result["confidence_scores"] == []


def test_uncertainty_node_with_mock_critiques(
    sample_insights: list[dict[str, Any]],
    large_clean_profile: dict[str, Any],
    mock_critique: dict[str, Any],
    mock_llm_scores: dict[str, Any],
) -> None:
    """Node correctly passes critiques to the estimator."""
    state: dict[str, Any] = {
        "insights": sample_insights,
        "critiques": [mock_critique],
        "profile_data": large_clean_profile,
    }
    with patch("agents.uncertainty.call_llm_with_messages") as mock_llm:
        mock_llm.return_value = json.dumps(mock_llm_scores)
        result = uncertainty_node(state)

    assert len(result["confidence_scores"]) == 2


# ---------------------------------------------------------------------------
# Integration test (requires running Ollama)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_estimate_live_llm(
    specific_insight: dict[str, Any],
    large_clean_profile: dict[str, Any],
    mock_critique: dict[str, Any],
) -> None:
    """Live LLM call: both LLM drivers must return integers 0-25."""
    estimator = UncertaintyEstimator()
    result = estimator.estimate(specific_insight, mock_critique, large_clean_profile)

    drivers = result["drivers"]
    for driver_key in ("statistical_evidence", "critic_assessment"):
        score = drivers[driver_key]["score"]
        assert isinstance(score, int), f"{driver_key} score must be int"
        assert 0 <= score <= 25, f"{driver_key} score {score} out of range"
