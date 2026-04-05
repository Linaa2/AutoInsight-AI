"""Compact LLM context builder for the profiler agent.

Converts a full :class:`~tools.profiler_engine.DataProfile` into a
token-efficient summary dict that is sent to the LLM instead of the raw
``profile.to_dict()`` payload.

Two detail modes are supported:

``fast`` (default)
    Absolute minimum per column — dtype, missing %, unique %, and a few
    type-specific highlights.  Global stats + deterministic flags give the
    LLM enough signal to write a focused report.  Typical payload ≈ 10 % of
    the full profile JSON.

``full``
    Extended per-column stats (quartiles, skewness, kurtosis, more top
    values) and a larger sample window.  Still smaller than the raw dump
    because redundant fields are omitted.

Usage::

    from agents.profiler_context import build_profiler_prompt_context

    ctx  = build_profiler_prompt_context(profile, detail_mode="fast")
    json_str = json.dumps(ctx, indent=2, default=str)

The returned dict has these top-level keys:
``dataset_overview``, ``data_quality``, ``column_summaries``,
``highlights``, ``sample_rows``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.profiler_engine import ColumnProfile, DataProfile

# ---------------------------------------------------------------------------
# Thresholds for deterministic flag generation
# ---------------------------------------------------------------------------

_HIGH_MISSING_PCT: float = 10.0  # % missing → "high missingness"
_HIGH_CARD_PCT: float = 50.0  # unique_pct → "high cardinality"
_HIGH_SKEW: float = 2.0  # |skewness| → "high skewness"
_HIGH_ZERO_PCT: float = 30.0  # zeros_pct → "high zero rate"
_NEAR_CONSTANT_UNIQUE: int = 3  # unique_count ≤ this → "near constant"
_SUSPICIOUS_UNIQUE_LOW: float = 0.5  # unique_pct < this for categorical → possibly an ID leak
_SUSPICIOUS_UNIQUE_HIGH: float = 95.0  # unique_pct > this for categorical → possible free-text/ID
_WIDE_DATE_DAYS: int = 365 * 5  # date_range_days → "wide date range"

# Fields included in fast vs full per-column summaries
_FAST_NUMERIC_FIELDS = ("min", "max", "mean", "missing_pct", "unique_pct", "zeros_pct")
_FULL_NUMERIC_FIELDS = (
    "min",
    "max",
    "mean",
    "median",
    "std",
    "q25",
    "q75",
    "skewness",
    "kurtosis",
    "missing_pct",
    "unique_pct",
    "zeros_pct",
    "zeros_count",
)

_FAST_CAT_TOP_N: int = 3
_FULL_CAT_TOP_N: int = 7


# ---------------------------------------------------------------------------
# Dataset-level quality rating
# ---------------------------------------------------------------------------


def _classify_quality_rating(profile: DataProfile) -> str:
    """Return a one-word quality label with emoji based on missing/dup rates."""
    missing_pct = profile.total_missing_pct
    dup_pct = profile.duplicates_pct
    if missing_pct < 5.0 and dup_pct < 2.0:
        return "✅ Good"
    if missing_pct < 20.0 and dup_pct < 10.0:
        return "⚠️ Fair"
    return "❌ Poor"


# ---------------------------------------------------------------------------
# Deterministic highlight extraction
# ---------------------------------------------------------------------------


def _extract_highlights(profile: DataProfile, detail_mode: str) -> list[str]:
    """Return a list of plain-English flag strings for notable data issues.

    Flags are fully deterministic — no LLM involved.
    """
    flags: list[str] = []

    for name, col in profile.columns.items():
        # --- Missing values ---
        if col.missing_pct > _HIGH_MISSING_PCT:
            flags.append(f"HIGH_MISSINGNESS: '{name}' is missing {col.missing_pct:.1f}% of values.")

        # --- Numeric-specific ---
        if col.dtype_category == "numeric":
            if col.skewness is not None and abs(col.skewness) > _HIGH_SKEW:
                direction = "right" if col.skewness > 0 else "left"
                flags.append(
                    f"HIGH_SKEWNESS: '{name}' is strongly {direction}-skewed "
                    f"(skewness={col.skewness:.2f})."
                )
            if col.zeros_pct is not None and col.zeros_pct > _HIGH_ZERO_PCT:
                flags.append(
                    f"HIGH_ZERO_RATE: '{name}' has {col.zeros_pct:.1f}% zero values — "
                    "check for default-fill or structural zeros."
                )
            if detail_mode == "full" and col.unique_count == 1:
                flags.append(f"CONSTANT_COLUMN: '{name}' has only one distinct value.")
            elif col.unique_count <= _NEAR_CONSTANT_UNIQUE and col.unique_count > 0:
                flags.append(
                    f"NEAR_CONSTANT: '{name}' has only {col.unique_count} distinct value(s)."
                )

        # --- Categorical / boolean ---
        if col.dtype_category in ("categorical", "boolean"):
            if col.unique_pct > _HIGH_CARD_PCT:
                flags.append(
                    f"HIGH_CARDINALITY: '{name}' has {col.unique_count} unique values "
                    f"({col.unique_pct:.1f}%) — may be a free-text or ID column."
                )
            elif col.unique_pct < _SUSPICIOUS_UNIQUE_LOW and detail_mode == "full":
                flags.append(
                    f"LOW_CARDINALITY: '{name}' has only {col.unique_count} category "
                    f"value(s) ({col.unique_pct:.1f}%) — consider one-hot encoding."
                )

        # --- Datetime ---
        if col.dtype_category == "datetime" and col.date_range_days is not None:
            if col.date_range_days > _WIDE_DATE_DAYS:
                flags.append(
                    f"WIDE_DATE_RANGE: '{name}' spans {col.date_range_days} days "
                    f"({col.date_range_days // 365} years) — verify intentional time window."
                )
            elif col.date_range_days == 0:
                flags.append(
                    f"ZERO_DATE_RANGE: '{name}' has all identical timestamps — "
                    "may be a constant/default date."
                )

    return flags


# ---------------------------------------------------------------------------
# Per-column summarisation
# ---------------------------------------------------------------------------


def _summarize_column(col: ColumnProfile, detail_mode: str) -> dict:
    """Return a compact dict with dtype-appropriate fields for *col*."""
    summary: dict = {
        "name": col.name,
        "dtype": col.dtype,
        "dtype_category": col.dtype_category,
        "missing_pct": col.missing_pct,
        "unique_count": col.unique_count,
        "unique_pct": col.unique_pct,
    }

    if col.dtype_category == "numeric":
        fields = _FULL_NUMERIC_FIELDS if detail_mode == "full" else _FAST_NUMERIC_FIELDS
        for f in fields:
            v = getattr(col, f, None)
            if v is not None:
                summary[f] = round(v, 4) if isinstance(v, float) else v

    elif col.dtype_category in ("categorical", "boolean"):
        top_n = _FULL_CAT_TOP_N if detail_mode == "full" else _FAST_CAT_TOP_N
        if col.top_values:
            top_items = sorted(col.top_values.items(), key=lambda x: -x[1])[:top_n]
            summary["top_values"] = dict(top_items)

    elif col.dtype_category == "datetime":
        summary["min_date"] = col.min_date
        summary["max_date"] = col.max_date
        summary["date_range_days"] = col.date_range_days

    return summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_profiler_prompt_context(
    profile: DataProfile,
    detail_mode: str = "fast",
) -> dict:
    """Build a compact, LLM-optimized context dict from a full :class:`DataProfile`.

    Args:
        profile:     The full deterministic profile produced by
                     :class:`~tools.profiler_engine.DataProfiler`.
        detail_mode: ``"fast"`` (default) for a minimal payload, or
                     ``"full"`` for extended column-level statistics.

    Returns:
        A JSON-serialisable dict with keys:
        ``dataset_overview``, ``data_quality``, ``column_summaries``,
        ``highlights``, ``sample_rows``.
    """
    mode = detail_mode if detail_mode in ("fast", "full") else "fast"

    # -- Dataset overview ---------------------------------------------------
    dataset_overview = {
        "rows": profile.shape[0],
        "columns": profile.shape[1],
        "memory_mb": profile.memory_mb,
        "column_type_counts": {
            "numeric": profile.numeric_cols,
            "categorical": profile.categorical_cols,
            "datetime": profile.datetime_cols,
            "boolean": profile.boolean_cols,
            "other": profile.other_cols,
        },
    }

    # -- Data quality -------------------------------------------------------
    data_quality = {
        "total_missing_pct": profile.total_missing_pct,
        "total_missing_count": profile.total_missing_count,
        "duplicates_count": profile.duplicates_count,
        "duplicates_pct": profile.duplicates_pct,
        "quality_rating": _classify_quality_rating(profile),
        "high_missing_columns": [
            {"column": col, "missing_pct": pct}
            for col, pct in sorted(profile.missing.items(), key=lambda x: -x[1])
            if pct > _HIGH_MISSING_PCT
        ],
    }

    # -- Column summaries ---------------------------------------------------
    column_summaries = [_summarize_column(col, mode) for col in profile.columns.values()]

    # -- Deterministic highlights -------------------------------------------
    highlights = _extract_highlights(profile, mode)

    # -- Sample rows (few rows in fast, more in full) -----------------------
    max_samples = 2 if mode == "fast" else 5
    sample_rows = profile.samples[:max_samples]

    return {
        "detail_mode": mode,
        "dataset_overview": dataset_overview,
        "data_quality": data_quality,
        "column_summaries": column_summaries,
        "highlights": highlights,
        "sample_rows": sample_rows,
    }
