"""Tests for compact context digests used by downstream LLM agents."""

from agents.context_digest import (
    build_confidence_digest,
    build_critiques_digest,
    build_insights_digest,
    build_profile_digest,
)
from agents.mock_profiler import get_mock_profile_data


def test_build_profile_digest_handles_legacy_profile_shape():
    profile = get_mock_profile_data()
    digest = build_profile_digest(profile, max_columns=3)

    assert "Dataset: 1500 rows x 8 columns" in digest
    assert "Duplicates:" in digest
    assert "Key columns:" in digest


def test_build_insights_digest_truncates_extra_items():
    insights = [
        {
            "title": f"Insight {idx}",
            "observation": "Observed pattern",
            "recommendation": "Act on it",
            "priority": "medium",
            "category": "general",
        }
        for idx in range(8)
    ]

    digest = build_insights_digest(insights, max_items=3)

    assert "Insight 0" in digest
    assert "Insight 2" in digest
    assert "... and 5 more insight(s)" in digest


def test_build_critiques_and_confidence_digests():
    critiques = [
        {
            "insight_title": "Sales concentration",
            "weaknesses": "Population mix not controlled.",
            "alternatives": "Warehouse location bias.",
            "confidence": "medium",
            "verdict": "partially_supported",
        }
    ]
    scores = [
        {
            "insight_title": "Sales concentration",
            "confidence_score": 68,
            "confidence_level": "medium",
            "summary": "Medium confidence (68%).",
        }
    ]

    critique_digest = build_critiques_digest(critiques)
    confidence_digest = build_confidence_digest(scores)

    assert "Sales concentration" in critique_digest
    assert "verdict=partially supported" in critique_digest
    assert "68% (medium)" in confidence_digest
