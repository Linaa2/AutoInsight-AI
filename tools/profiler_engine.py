"""Deterministic dataset profiling using pandas.

Computes shape, column statistics, missing values, duplicates, value
distributions, and sample rows — all without any LLM involvement.
"""

import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_module_logger

logger = get_module_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    """Statistics for a single DataFrame column."""

    name: str
    dtype: str
    dtype_category: str  # numeric | categorical | datetime | boolean | other
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float

    # --- Numeric ---
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    q25: float | None = None
    q75: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zeros_count: int | None = None
    zeros_pct: float | None = None

    # --- Categorical / Boolean ---
    top_values: dict[str, int] | None = None

    # --- Datetime ---
    min_date: str | None = None
    max_date: str | None = None
    date_range_days: int | None = None


@dataclass
class DataProfile:
    """Full deterministic profile of a DataFrame."""

    shape: list[int]  # [rows, cols]
    duplicates_count: int
    duplicates_pct: float
    total_missing_count: int
    total_missing_pct: float
    memory_mb: float
    numeric_cols: int
    categorical_cols: int
    datetime_cols: int
    boolean_cols: int
    other_cols: int
    columns: dict[str, ColumnProfile]
    missing: dict[str, float]  # col -> missing %
    samples: list[dict[str, Any]]
    created_at: str  # ISO-8601 UTC timestamp

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict matching the required output schema:

        {
            "shape":   [rows, cols],
            "columns": {col_name: {...}},
            "missing": {col_name: missing_pct},
            "stats":   {duplicates, memory, column type counts, ...},
            "samples": [{...}, ...]
        }
        """
        d = asdict(self)
        # Group all global statistics under a dedicated "stats" key
        stats_keys = [
            "duplicates_count",
            "duplicates_pct",
            "total_missing_count",
            "total_missing_pct",
            "memory_mb",
            "numeric_cols",
            "categorical_cols",
            "datetime_cols",
            "boolean_cols",
            "other_cols",
            "created_at",
        ]
        d["stats"] = {k: d.pop(k) for k in stats_keys if k in d}
        return d


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class DataProfiler:
    """Computes a deterministic :class:`DataProfile` for any pandas DataFrame.

    Environment variables:
        PROFILER_SAMPLE_ROWS:  Number of sample rows to include (default 5).
        PROFILER_TOP_VALUES:   Max number of top values for categorical/boolean
                               columns (default 10).
    """

    def __init__(self) -> None:
        self._sample_rows = int(os.getenv("PROFILER_SAMPLE_ROWS", "5"))
        self._top_values = int(os.getenv("PROFILER_TOP_VALUES", "10"))

    def profile(self, df: pd.DataFrame) -> DataProfile:
        """Compute a full profile for *df* and return a :class:`DataProfile`."""
        n_rows, n_cols = df.shape
        logger.info("Profiling DataFrame — %d rows, %d cols", n_rows, n_cols)
        t0 = time.perf_counter()

        duplicates_count = int(df.duplicated().sum())
        missing_per_col = df.isnull().sum()
        total_missing = int(missing_per_col.sum())
        memory_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 4)

        columns: dict[str, ColumnProfile] = {}
        dtype_counts = {
            "numeric": 0,
            "categorical": 0,
            "datetime": 0,
            "boolean": 0,
            "other": 0,
        }

        for col in df.columns:
            cp = self._profile_column(df[col])
            columns[col] = cp
            dtype_counts[cp.dtype_category] += 1

        missing = {
            col: round(float(missing_per_col[col]) / n_rows * 100, 2) if n_rows > 0 else 0.0
            for col in df.columns
        }

        # Replace NaN/NaT with None for JSON safety
        samples = df.head(self._sample_rows).replace({np.nan: None}).to_dict(orient="records")
        # Ensure all keys are strings
        samples = [{str(k): v for k, v in row.items()} for row in samples]

        result = DataProfile(
            shape=[n_rows, n_cols],
            duplicates_count=duplicates_count,
            duplicates_pct=round(duplicates_count / n_rows * 100, 2) if n_rows > 0 else 0.0,
            total_missing_count=total_missing,
            total_missing_pct=(
                round(total_missing / (n_rows * n_cols) * 100, 2) if n_rows * n_cols > 0 else 0.0
            ),
            memory_mb=memory_mb,
            numeric_cols=dtype_counts["numeric"],
            categorical_cols=dtype_counts["categorical"],
            datetime_cols=dtype_counts["datetime"],
            boolean_cols=dtype_counts["boolean"],
            other_cols=dtype_counts["other"],
            columns=columns,
            missing=missing,
            samples=samples,
            created_at=datetime.now(UTC).isoformat(),
        )
        logger.info("Profile complete in %.2f s", time.perf_counter() - t0)
        return result

    # ------------------------------------------------------------------
    # Column-level profiling
    # ------------------------------------------------------------------

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        n = len(series)
        missing_count = int(series.isnull().sum())
        unique_count = int(series.nunique())

        base: dict[str, Any] = {
            "name": str(series.name),
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_pct": round(missing_count / n * 100, 2) if n > 0 else 0.0,
            "unique_count": unique_count,
            "unique_pct": round(unique_count / n * 100, 2) if n > 0 else 0.0,
        }

        if pd.api.types.is_bool_dtype(series):
            return ColumnProfile(
                **base,
                dtype_category="boolean",
                top_values=self._value_counts(series),
            )

        if pd.api.types.is_numeric_dtype(series):
            return self._profile_numeric(series, base)

        if pd.api.types.is_datetime64_any_dtype(series):
            return self._profile_datetime(series, base)

        # Default: categorical / object
        return ColumnProfile(
            **base,
            dtype_category="categorical",
            top_values=self._value_counts(series),
        )

    def _profile_numeric(self, series: pd.Series, base: dict[str, Any]) -> ColumnProfile:
        desc = series.describe()
        n = len(series)
        zeros_count = int((series == 0).sum())
        return ColumnProfile(
            **base,
            dtype_category="numeric",
            min=self._safe_float(desc.get("min")),
            max=self._safe_float(desc.get("max")),
            mean=self._safe_float(desc.get("mean")),
            median=self._safe_float(series.median()),
            std=self._safe_float(desc.get("std")),
            q25=self._safe_float(desc.get("25%")),
            q75=self._safe_float(desc.get("75%")),
            skewness=self._safe_float(series.skew()),
            kurtosis=self._safe_float(series.kurtosis()),
            zeros_count=zeros_count,
            zeros_pct=round(zeros_count / n * 100, 2) if n > 0 else 0.0,
        )

    def _profile_datetime(self, series: pd.Series, base: dict[str, Any]) -> ColumnProfile:
        valid = series.dropna()
        date_range_days = None
        min_date = max_date = None
        if len(valid) > 0:
            min_date = str(valid.min())
            max_date = str(valid.max())
            date_range_days = int((valid.max() - valid.min()).days)
        return ColumnProfile(
            **base,
            dtype_category="datetime",
            min_date=min_date,
            max_date=max_date,
            date_range_days=date_range_days,
        )

    def _value_counts(self, series: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts().head(self._top_values).items()}

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert *value* to float, returning None for NaN/Inf/errors."""
        try:
            f = float(value)
            return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
        except (TypeError, ValueError):
            return None
