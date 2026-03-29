"""Tests for visualization.parser.

Covers the four key behaviors:
    1. Valid JSON → returns specs with no error
    2. Valid JSON inside markdown fences → fences are stripped before parsing
    3. Malformed / non-JSON output → fails gracefully (no exception, clear error)
    4. Missing required field → chart is rejected with a descriptive error
    5. Unknown chart_type → chart is rejected and reported in error string
"""

from __future__ import annotations

import json

import pytest

from visualization.parser import parse_llm_output
from visualization.schemas import ALLOWED_CHART_TYPES

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

_VALID_SPEC: dict = {
    "charts": [
        {
            "title": "Sales by Region",
            "chart_type": "bar",
            "code": "fig = px.bar(df, x='region', y='sales', title='Sales by Region')",
            "explanation": "Shows how sales are distributed across regions.",
            "columns_used": ["region", "sales"],
        }
    ]
}

_VALID_JSON = json.dumps(_VALID_SPEC)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_parse_valid_json_returns_specs() -> None:
    """Parser succeeds on well-formed raw JSON output."""
    charts, error = parse_llm_output(_VALID_JSON)

    assert error is None
    assert len(charts) == 1
    assert charts[0]["title"] == "Sales by Region"
    assert charts[0]["chart_type"] == "bar"
    assert "code" in charts[0]


def test_parse_valid_json_with_markdown_fences() -> None:
    """Parser strips ```json ... ``` fences before parsing."""
    raw = f"```json\n{_VALID_JSON}\n```"
    charts, error = parse_llm_output(raw)

    assert error is None
    assert len(charts) == 1


def test_parse_valid_json_with_plain_fences() -> None:
    """Parser strips plain ``` ... ``` fences (without the json hint)."""
    raw = f"```\n{_VALID_JSON}\n```"
    charts, error = parse_llm_output(raw)

    assert error is None
    assert len(charts) == 1


def test_parse_optional_fields_are_passed_through() -> None:
    """Optional fields (explanation, columns_used) are preserved when present."""
    charts, error = parse_llm_output(_VALID_JSON)

    assert error is None
    spec = charts[0]
    assert spec.get("explanation") == "Shows how sales are distributed across regions."
    assert spec.get("columns_used") == ["region", "sales"]


def test_parse_spec_without_optional_fields() -> None:
    """Specs with only required fields are accepted."""
    minimal = json.dumps(
        {
            "charts": [
                {
                    "title": "Histogram",
                    "chart_type": "histogram",
                    "code": "fig = px.histogram(df, x='age')",
                }
            ]
        }
    )
    charts, error = parse_llm_output(minimal)

    assert error is None
    assert len(charts) == 1
    assert "explanation" not in charts[0]


def test_parse_all_allowed_chart_types() -> None:
    """Every chart type in ALLOWED_CHART_TYPES is accepted by the parser."""
    for chart_type in ALLOWED_CHART_TYPES:
        payload = json.dumps(
            {
                "charts": [
                    {
                        "title": "Test",
                        "chart_type": chart_type,
                        "code": "fig = px.bar(df)",
                    }
                ]
            }
        )
        charts, error = parse_llm_output(payload)
        assert len(charts) == 1, f"Expected chart for type {chart_type!r}, got error={error}"


# ---------------------------------------------------------------------------
# Failure / graceful-degradation tests
# ---------------------------------------------------------------------------


def test_parse_malformed_json_returns_error() -> None:
    """Non-JSON output does not crash — returns empty list and error string."""
    raw = "Sorry, I cannot help with that dataset."
    charts, error = parse_llm_output(raw)

    assert charts == []
    assert error is not None
    assert len(error) > 0


def test_parse_truncated_json_returns_error() -> None:
    """Truncated JSON (incomplete object) does not crash."""
    raw = '{"charts": [{"title": "Broken'  # truncated
    charts, error = parse_llm_output(raw)

    assert charts == []
    assert error is not None


def test_parse_missing_code_field_rejects_chart() -> None:
    """A chart spec missing the 'code' field is rejected with a clear error."""
    payload = json.dumps(
        {"charts": [{"title": "No code", "chart_type": "bar"}]}  # missing 'code'
    )
    charts, error = parse_llm_output(payload)

    assert charts == []
    assert error is not None
    assert "code" in error.lower() or "missing" in error.lower()


def test_parse_missing_title_field_rejects_chart() -> None:
    """A chart spec missing 'title' is rejected."""
    payload = json.dumps({"charts": [{"chart_type": "bar", "code": "fig = px.bar(df)"}]})
    charts, error = parse_llm_output(payload)

    assert charts == []
    assert error is not None


def test_parse_invalid_chart_type_rejects_chart() -> None:
    """An unknown chart_type is rejected (not silently ignored)."""
    payload = json.dumps(
        {
            "charts": [
                {
                    "title": "Radar",
                    "chart_type": "radar",  # not in ALLOWED_CHART_TYPES
                    "code": "fig = go.Figure()",
                }
            ]
        }
    )
    charts, error = parse_llm_output(payload)

    assert charts == []
    assert error is not None
    assert "radar" in error.lower() or "unknown" in error.lower()


def test_parse_partial_success_mixed_specs() -> None:
    """Valid specs are returned even when some other specs in the same JSON fail."""
    payload = json.dumps(
        {
            "charts": [
                # valid
                {
                    "title": "Good chart",
                    "chart_type": "bar",
                    "code": "fig = px.bar(df)",
                },
                # invalid — missing 'code'
                {"title": "Bad chart", "chart_type": "line"},
            ]
        }
    )
    charts, error = parse_llm_output(payload)

    # At least the valid chart should come through
    assert len(charts) == 1
    assert charts[0]["title"] == "Good chart"
    # Error should mention the failed spec
    assert error is not None


def test_parse_json_nested_in_prose() -> None:
    """Parser extracts JSON when surrounded by explanatory prose."""
    prose_prefix = "Sure! Here are your charts:\n\n"
    prose_suffix = "\n\nLet me know if you need more."
    raw = prose_prefix + _VALID_JSON + prose_suffix

    charts, error = parse_llm_output(raw)

    assert error is None
    assert len(charts) == 1


# ---------------------------------------------------------------------------
# Guard: make sure the test module itself imports without error
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_import_parser() -> None:
    """Smoke test: the parser module is importable."""
    from visualization import parse_llm_output as _fn

    assert callable(_fn)
