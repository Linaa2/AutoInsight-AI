"""Unit tests for evaluation/validators.py."""

import pytest

from evaluation.validators import (
    check_column_references,
    check_insight_fields,
    check_required_sections,
    check_statistic_accuracy,
)
from tools.profiler_engine import DataProfiler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profile(sample_df):
    """DataProfile produced from the shared sample_df fixture."""
    return DataProfiler().profile(sample_df)


# ---------------------------------------------------------------------------
# check_column_references
# ---------------------------------------------------------------------------


def test_check_column_references_valid(profile):
    text = "The column 'age' shows a wide range of values."
    result = check_column_references(text, profile)
    assert "age" in result["present"]


def test_check_column_references_invalid(profile):
    """A fabricated column name should NOT appear in 'present'."""
    text = "The column 'xyz_not_real' shows values."
    result = check_column_references(text, profile)
    assert "xyz_not_real" not in result["present"]


def test_check_column_references_empty_text(profile):
    result = check_column_references("", profile)
    assert len(result["unmentioned"]) == len(profile.columns)
    assert result["present"] == []


def test_check_column_references_all_mentioned(profile):
    text = " ".join(profile.columns.keys())
    result = check_column_references(text, profile)
    assert set(result["present"]) == set(profile.columns.keys())
    assert result["unmentioned"] == []


# ---------------------------------------------------------------------------
# check_statistic_accuracy
# ---------------------------------------------------------------------------


def test_check_statistic_accuracy_within_tolerance(profile):
    row_count = profile.shape[0]
    text = f"The dataset has {row_count} rows."
    result = check_statistic_accuracy(text, profile, tolerance=0.05)
    assert float(row_count) in result["accurate"]


def test_check_statistic_accuracy_outside_tolerance(profile):
    text = "The dataset has 9999999 rows."
    result = check_statistic_accuracy(text, profile, tolerance=0.05)
    assert 9999999.0 in result["inaccurate"]


def test_check_statistic_accuracy_no_numbers(profile):
    result = check_statistic_accuracy("No numbers here at all.", profile)
    assert result["accurate"] == []
    assert result["inaccurate"] == []


def test_check_statistic_accuracy_returns_known_values_count(profile):
    result = check_statistic_accuracy("", profile)
    assert result["known_values_count"] > 0


# ---------------------------------------------------------------------------
# check_required_sections
# ---------------------------------------------------------------------------


def test_check_required_sections_all_present():
    text = (
        "## Dataset Overview\ncontent\n"
        "## Data Quality Assessment\ncontent\n"
        "## Column-by-Column Analysis\ncontent\n"
        "## Statistical Highlights\ncontent\n"
        "## Key Takeaways\ncontent\n"
    )
    result = check_required_sections(text, "profiler")
    assert result["missing"] == [], f"Expected no missing sections, got: {result['missing']}"


def test_check_required_sections_some_missing():
    text = "## Dataset Overview\nsome content"
    result = check_required_sections(text, "profiler")
    assert "Data Quality Assessment" in result["missing"]
    assert "Column-by-Column Analysis" in result["missing"]


def test_check_required_sections_unknown_type():
    result = check_required_sections("anything", "unknown_type")
    assert result["missing"] == []
    assert result["present"] == []


def test_check_required_sections_reporter():
    text = (
        "## Executive Summary\n"
        "## Dataset Description\n"
        "## Key Insights\n"
        "## Visualizations\n"
        "## Recommendations\n"
        "## Limitations & Next Steps\n"
    )
    result = check_required_sections(text, "reporter")
    assert result["missing"] == []


# ---------------------------------------------------------------------------
# check_insight_fields
# ---------------------------------------------------------------------------


def _valid_insight(title="High concentration"):
    return {
        "title": title,
        "observation": "32% of orders come from Ile-de-France.",
        "hypothesis": "Population density drives this.",
        "recommendation": "Expand logistics into other regions.",
        "priority": "high",
    }


def test_check_insight_fields_valid():
    insights = [_valid_insight(), _valid_insight("Another insight")]
    result = check_insight_fields(insights)
    assert result["valid_count"] == 2
    assert result["issues"] == []


def test_check_insight_fields_missing_key():
    bad = _valid_insight()
    del bad["recommendation"]
    result = check_insight_fields([bad])
    assert result["valid_count"] == 0
    assert len(result["issues"]) == 1
    assert any("recommendation" in p for p in result["issues"][0]["problems"])


def test_check_insight_fields_empty_value():
    bad = _valid_insight()
    bad["title"] = ""
    result = check_insight_fields([bad])
    assert result["valid_count"] == 0
    assert any("title" in p for p in result["issues"][0]["problems"])


def test_check_insight_fields_empty_list():
    result = check_insight_fields([])
    assert result["valid_count"] == 0
    assert result["issues"] == []


def test_check_insight_fields_mixed():
    insights = [
        _valid_insight(),
        {
            "title": "",
            "observation": "x",
            "hypothesis": "y",
            "recommendation": "z",
            "priority": "low",
        },
    ]
    result = check_insight_fields(insights)
    assert result["valid_count"] == 1
    assert len(result["issues"]) == 1
