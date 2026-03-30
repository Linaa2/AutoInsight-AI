"""Tests for tools.profiler_engine."""

import json

import pandas as pd
import pytest

from tools.profiler_engine import DataProfile, DataProfiler


class TestDataProfiler:
    # ------------------------------------------------------------------
    # Return types
    # ------------------------------------------------------------------

    def test_profile_returns_data_profile(self, sample_df):
        assert isinstance(DataProfiler().profile(sample_df), DataProfile)

    # ------------------------------------------------------------------
    # Shape & global stats
    # ------------------------------------------------------------------

    def test_shape(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        assert profile.shape == list(sample_df.shape)

    def test_columns_coverage(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        assert set(profile.columns.keys()) == set(sample_df.columns)

    def test_duplicates_detection(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
        profile = DataProfiler().profile(df)
        assert profile.duplicates_count == 1
        assert profile.duplicates_pct == pytest.approx(100 / 3, rel=1e-3)

    def test_no_duplicates(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        assert profile.duplicates_count == 0

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------

    def test_missing_count_per_column(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        # fixture: 'age' has 2 None values
        assert profile.columns["age"].missing_count == 2
        assert profile.missing["age"] > 0

    def test_total_missing_pct_is_nonzero(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        assert profile.total_missing_pct > 0

    def test_no_missing_values(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        profile = DataProfiler().profile(df)
        assert profile.total_missing_count == 0
        assert all(v == 0.0 for v in profile.missing.values())

    # ------------------------------------------------------------------
    # dtype_category detection
    # ------------------------------------------------------------------

    def test_numeric_dtype_category(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        assert profile.columns["salary"].dtype_category == "numeric"
        assert profile.columns["age"].dtype_category == "numeric"

    def test_categorical_dtype_category(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        assert profile.columns["department"].dtype_category == "categorical"

    def test_boolean_dtype_category(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        assert profile.columns["is_active"].dtype_category == "boolean"

    def test_datetime_dtype_category(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"])})
        profile = DataProfiler().profile(df)
        assert profile.columns["ts"].dtype_category == "datetime"

    # ------------------------------------------------------------------
    # Numeric column statistics
    # ------------------------------------------------------------------

    def test_numeric_stats_present(self, sample_df):
        cp = DataProfiler().profile(sample_df).columns["salary"]
        for attr in ("min", "max", "mean", "median", "std", "q25", "q75"):
            assert getattr(cp, attr) is not None, f"'{attr}' should not be None"

    def test_skewness_and_kurtosis(self, sample_df):
        cp = DataProfiler().profile(sample_df).columns["salary"]
        assert cp.skewness is not None
        assert cp.kurtosis is not None

    def test_zeros_count(self):
        df = pd.DataFrame({"v": [0, 0, 1, 2, 3]})
        cp = DataProfiler().profile(df).columns["v"]
        assert cp.zeros_count == 2
        assert cp.zeros_pct == pytest.approx(40.0)

    # ------------------------------------------------------------------
    # Categorical column statistics
    # ------------------------------------------------------------------

    def test_top_values_present(self, sample_df):
        cp = DataProfiler().profile(sample_df).columns["department"]
        assert cp.top_values is not None
        assert isinstance(cp.top_values, dict)
        assert len(cp.top_values) > 0

    def test_top_values_respects_env(self, sample_df):
        cp = DataProfiler(top_values=2).profile(sample_df).columns["department"]
        assert len(cp.top_values) <= 2

    # ------------------------------------------------------------------
    # Datetime column statistics
    # ------------------------------------------------------------------

    def test_datetime_range(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"])})
        cp = DataProfiler().profile(df).columns["ts"]
        assert cp.min_date is not None
        assert cp.max_date is not None
        assert cp.date_range_days == 365

    # ------------------------------------------------------------------
    # Samples
    # ------------------------------------------------------------------

    def test_sample_count_default(self, sample_df):
        profile = DataProfiler(sample_rows=5).profile(sample_df)
        assert len(profile.samples) == 5

    def test_sample_count_custom(self, sample_df):
        profile = DataProfiler(sample_rows=3).profile(sample_df)
        assert len(profile.samples) == 3

    def test_samples_have_string_keys(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        for row in profile.samples:
            assert all(isinstance(k, str) for k in row)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def test_to_dict_is_json_serialisable(self, sample_df):
        profile = DataProfiler().profile(sample_df)
        # Should not raise
        json.dumps(profile.to_dict(), default=str)

    def test_to_dict_shape(self, sample_df):
        d = DataProfiler().profile(sample_df).to_dict()
        assert "shape" in d
        assert d["shape"] == list(sample_df.shape)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        profile = DataProfiler().profile(df)
        assert profile.shape == [0, 1]
        assert profile.duplicates_pct == 0.0

    def test_all_missing_column(self):
        df = pd.DataFrame({"x": [None, None, None]})
        cp = DataProfiler().profile(df).columns["x"]
        assert cp.missing_count == 3
        assert cp.missing_pct == pytest.approx(100.0)
