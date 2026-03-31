"""Tests for agents.profiler_context — compact LLM context builder."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from agents.profiler_context import (
    _classify_quality_rating,
    _extract_highlights,
    _summarize_column,
    build_profiler_prompt_context,
)
from tools.profiler_engine import DataProfiler

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profile(sample_df):
    """Full DataProfile computed from the shared sample fixture."""
    return DataProfiler().profile(sample_df)


@pytest.fixture
def high_missing_df():
    """DataFrame with severe missingness to trigger HIGH_MISSINGNESS flags."""
    return pd.DataFrame(
        {
            "id": range(10),
            "gap_col": [None] * 8 + [1.0, 2.0],  # 80 % missing
            "ok_col": range(10),
        }
    )


@pytest.fixture
def skewed_df():
    """DataFrame with a heavily right-skewed numeric column."""
    values = [0.01] * 90 + [
        1000.0,
        2000.0,
        3000.0,
        4000.0,
        5000.0,
        6000.0,
        7000.0,
        8000.0,
        9000.0,
        10000.0,
    ]
    return pd.DataFrame({"skewed": values})


@pytest.fixture
def high_card_df():
    """DataFrame with a high-cardinality categorical column."""
    return pd.DataFrame({"uid": [f"user_{i}" for i in range(100)]})


@pytest.fixture
def datetime_df():
    """DataFrame with a datetime column spanning several years."""
    dates = pd.date_range("2015-01-01", "2024-12-31", periods=50)
    return pd.DataFrame({"event_date": dates})


# ---------------------------------------------------------------------------
# Part A: Top-level structure
# ---------------------------------------------------------------------------


class TestBuildContextStructure:
    """build_profiler_prompt_context returns the correct top-level keys."""

    def test_fast_mode_has_required_keys(self, profile):
        ctx = build_profiler_prompt_context(profile, "fast")
        required = {
            "detail_mode",
            "dataset_overview",
            "data_quality",
            "column_summaries",
            "highlights",
            "sample_rows",
        }
        assert required <= ctx.keys()

    def test_full_mode_has_required_keys(self, profile):
        ctx = build_profiler_prompt_context(profile, "full")
        required = {
            "detail_mode",
            "dataset_overview",
            "data_quality",
            "column_summaries",
            "highlights",
            "sample_rows",
        }
        assert required <= ctx.keys()

    def test_fast_mode_label(self, profile):
        ctx = build_profiler_prompt_context(profile, "fast")
        assert ctx["detail_mode"] == "fast"

    def test_full_mode_label(self, profile):
        ctx = build_profiler_prompt_context(profile, "full")
        assert ctx["detail_mode"] == "full"

    def test_unknown_mode_falls_back_to_fast(self, profile):
        ctx = build_profiler_prompt_context(profile, "turbo")
        assert ctx["detail_mode"] == "fast"

    def test_dataset_overview_keys(self, profile):
        ov = build_profiler_prompt_context(profile, "fast")["dataset_overview"]
        assert "rows" in ov and "columns" in ov and "memory_mb" in ov
        assert "column_type_counts" in ov

    def test_data_quality_keys(self, profile):
        dq = build_profiler_prompt_context(profile, "fast")["data_quality"]
        for key in (
            "total_missing_pct",
            "duplicates_count",
            "quality_rating",
            "high_missing_columns",
        ):
            assert key in dq, f"Missing key: {key}"

    def test_column_summaries_length(self, profile, sample_df):
        ctx = build_profiler_prompt_context(profile, "fast")
        assert len(ctx["column_summaries"]) == len(sample_df.columns)

    def test_sample_rows_fast_capped(self, profile):
        ctx = build_profiler_prompt_context(profile, "fast")
        assert len(ctx["sample_rows"]) <= 2

    def test_sample_rows_full_larger(self, profile):
        ctx = build_profiler_prompt_context(profile, "full")
        assert len(ctx["sample_rows"]) <= 5


# ---------------------------------------------------------------------------
# Part B: Payload size comparison
# ---------------------------------------------------------------------------


class TestPayloadSize:
    """Fast mode JSON must be measurably smaller than full mode JSON."""

    def test_fast_smaller_than_full(self, profile):
        fast_size = len(json.dumps(build_profiler_prompt_context(profile, "fast"), default=str))
        full_size = len(json.dumps(build_profiler_prompt_context(profile, "full"), default=str))
        assert fast_size < full_size, "fast payload should be smaller than full payload"

    def test_fast_significantly_smaller_than_raw(self, profile):
        """Fast context should be markedly smaller than profile.to_dict() JSON."""
        raw_size = len(json.dumps(profile.to_dict(), indent=2, default=str))
        fast_size = len(json.dumps(build_profiler_prompt_context(profile, "fast"), default=str))
        assert fast_size < raw_size, "fast context should be smaller than raw profile JSON"


# ---------------------------------------------------------------------------
# Part C: Column summary fields per dtype
# ---------------------------------------------------------------------------


class TestColumnSummaryFields:
    """_summarize_column returns the right fields for each dtype_category."""

    def test_numeric_summary_has_min_max_mean(self, profile):
        col = profile.columns["salary"]
        s = _summarize_column(col, "fast")
        assert "min" in s and "max" in s and "mean" in s

    def test_numeric_full_has_quartiles(self, profile):
        col = profile.columns["salary"]
        s = _summarize_column(col, "full")
        assert "q25" in s and "q75" in s

    def test_numeric_fast_no_quartiles(self, profile):
        col = profile.columns["salary"]
        s = _summarize_column(col, "fast")
        assert "q25" not in s and "q75" not in s

    def test_numeric_full_has_skewness(self, profile):
        col = profile.columns["salary"]
        s = _summarize_column(col, "full")
        assert "skewness" in s

    def test_categorical_summary_has_top_values(self, profile):
        col = profile.columns["department"]
        s = _summarize_column(col, "fast")
        assert "top_values" in s

    def test_categorical_fast_top_values_capped_at_3(self, profile):
        col = profile.columns["department"]
        s = _summarize_column(col, "fast")
        assert len(s["top_values"]) <= 3

    def test_categorical_full_top_values_up_to_7(self, profile):
        col = profile.columns["department"]
        s = _summarize_column(col, "full")
        assert len(s["top_values"]) <= 7

    def test_boolean_summary_has_top_values(self, profile):
        col = profile.columns["is_active"]
        s = _summarize_column(col, "fast")
        assert "top_values" in s

    def test_datetime_summary_has_date_fields(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2020-01-01", "2022-06-15", "2024-12-31"])})
        p = DataProfiler().profile(df)
        col = p.columns["ts"]
        s = _summarize_column(col, "fast")
        assert "min_date" in s and "max_date" in s and "date_range_days" in s

    def test_every_summary_has_base_fields(self, profile):
        for col in profile.columns.values():
            s = _summarize_column(col, "fast")
            assert "name" in s
            assert "dtype_category" in s
            assert "missing_pct" in s


# ---------------------------------------------------------------------------
# Part D: Deterministic highlights
# ---------------------------------------------------------------------------


class TestExtractHighlights:
    """_extract_highlights fires the right flags."""

    def test_high_missingness_flag(self, high_missing_df):
        p = DataProfiler().profile(high_missing_df)
        flags = _extract_highlights(p, "fast")
        flagged_cols = [f for f in flags if "HIGH_MISSINGNESS" in f and "'gap_col'" in f]
        assert flagged_cols, f"Expected HIGH_MISSINGNESS for gap_col, got: {flags}"

    def test_no_flag_for_low_missingness(self, profile):
        flags = _extract_highlights(profile, "fast")
        # salary has 0 % missing — should not produce HIGH_MISSINGNESS
        assert not any("HIGH_MISSINGNESS" in f and "'salary'" in f for f in flags)

    def test_high_skewness_flag(self, skewed_df):
        p = DataProfiler().profile(skewed_df)
        flags = _extract_highlights(p, "fast")
        assert any("HIGH_SKEWNESS" in f for f in flags), f"Expected HIGH_SKEWNESS, got: {flags}"

    def test_high_cardinality_flag(self, high_card_df):
        p = DataProfiler().profile(high_card_df)
        flags = _extract_highlights(p, "fast")
        assert any(
            "HIGH_CARDINALITY" in f for f in flags
        ), f"Expected HIGH_CARDINALITY, got: {flags}"

    def test_wide_date_range_flag(self, datetime_df):
        p = DataProfiler().profile(datetime_df)
        flags = _extract_highlights(p, "fast")
        assert any("WIDE_DATE_RANGE" in f for f in flags), f"Expected WIDE_DATE_RANGE, got: {flags}"

    def test_no_flags_for_clean_df(self):
        """A clean, low-cardinality, no-missing DataFrame should produce zero flags."""
        df = pd.DataFrame(
            {
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "category": ["A", "B", "A", "B", "A"],
            }
        )
        p = DataProfiler().profile(df)
        flags = _extract_highlights(p, "fast")
        assert flags == [], f"Expected no flags for clean df, got: {flags}"

    def test_high_missing_columns_in_data_quality(self, high_missing_df):
        """build_profiler_prompt_context populates high_missing_columns correctly."""
        p = DataProfiler().profile(high_missing_df)
        ctx = build_profiler_prompt_context(p, "fast")
        hmc = ctx["data_quality"]["high_missing_columns"]
        col_names = [entry["column"] for entry in hmc]
        assert "gap_col" in col_names


# ---------------------------------------------------------------------------
# Part E: Quality rating
# ---------------------------------------------------------------------------


class TestQualityRating:
    def test_good_rating_for_clean_data(self):
        df = pd.DataFrame({"x": range(100), "y": range(100)})
        p = DataProfiler().profile(df)
        assert _classify_quality_rating(p) == "✅ Good"

    def test_poor_rating_for_highly_missing(self, high_missing_df):
        p = DataProfiler().profile(high_missing_df)
        rating = _classify_quality_rating(p)
        assert rating in ("⚠️ Fair", "❌ Poor")

    def test_rating_appears_in_context(self, profile):
        ctx = build_profiler_prompt_context(profile, "fast")
        assert ctx["data_quality"]["quality_rating"] in ("✅ Good", "⚠️ Fair", "❌ Poor")


# ---------------------------------------------------------------------------
# Part F: Dataset overview accuracy
# ---------------------------------------------------------------------------


class TestDatasetOverview:
    def test_rows_and_cols_match_profile(self, profile, sample_df):
        ctx = build_profiler_prompt_context(profile, "fast")
        ov = ctx["dataset_overview"]
        assert ov["rows"] == sample_df.shape[0]
        assert ov["columns"] == sample_df.shape[1]

    def test_type_counts_are_non_negative(self, profile):
        counts = build_profiler_prompt_context(profile, "fast")["dataset_overview"][
            "column_type_counts"
        ]
        for k, v in counts.items():
            assert v >= 0, f"Negative count for {k}: {v}"

    def test_type_counts_sum_to_total_columns(self, profile, sample_df):
        counts = build_profiler_prompt_context(profile, "fast")["dataset_overview"][
            "column_type_counts"
        ]
        assert sum(counts.values()) == sample_df.shape[1]
