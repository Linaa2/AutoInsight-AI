"""Unit tests for evaluation/rubrics.py."""

import pytest

from evaluation.rubrics import REQUIRED_SECTIONS, RUBRICS

# ---------------------------------------------------------------------------
# Weight integrity
# ---------------------------------------------------------------------------


def test_profiler_rubric_weights_sum_to_one():
    total = sum(c["weight"] for c in RUBRICS["profiler"])
    assert abs(total - 1.0) < 1e-9


def test_analyst_rubric_weights_sum_to_one():
    total = sum(c["weight"] for c in RUBRICS["analyst"])
    assert abs(total - 1.0) < 1e-9


def test_reporter_rubric_weights_sum_to_one():
    total = sum(c["weight"] for c in RUBRICS["reporter"])
    assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Required keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("artifact_type", ["profiler", "analyst", "reporter"])
def test_all_rubric_entries_have_required_keys(artifact_type):
    for entry in RUBRICS[artifact_type]:
        assert "name" in entry, f"Missing 'name' in {artifact_type} rubric entry: {entry}"
        assert "label" in entry, f"Missing 'label' in {artifact_type} rubric entry: {entry}"
        assert "weight" in entry, f"Missing 'weight' in {artifact_type} rubric entry: {entry}"


# ---------------------------------------------------------------------------
# Weight bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("artifact_type", ["profiler", "analyst", "reporter"])
def test_weight_is_float_between_zero_and_one(artifact_type):
    for entry in RUBRICS[artifact_type]:
        w = entry["weight"]
        assert isinstance(w, float), f"Weight not a float in {artifact_type}: {entry}"
        assert 0.0 < w <= 1.0, f"Weight out of range in {artifact_type}: {entry}"


# ---------------------------------------------------------------------------
# REQUIRED_SECTIONS
# ---------------------------------------------------------------------------


def test_required_sections_profiler_non_empty():
    assert len(REQUIRED_SECTIONS["profiler"]) > 0


def test_required_sections_reporter_non_empty():
    assert len(REQUIRED_SECTIONS["reporter"]) > 0
