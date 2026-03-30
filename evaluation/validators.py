"""Deterministic validators for evaluation pre-checks.

These run before the LLM judge and inject factual ground-truth into the
judge prompt so the LLM does not need to re-derive facts it cannot reliably
verify on its own.

All functions are pure (no LLM calls, no side effects).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from evaluation.rubrics import REQUIRED_SECTIONS

if TYPE_CHECKING:
    from tools.profiler_engine import DataProfile

# Required insight fields (non-empty strings)
_INSIGHT_REQUIRED_FIELDS = ("title", "observation", "hypothesis", "recommendation", "priority")


# ---------------------------------------------------------------------------
# check_column_references
# ---------------------------------------------------------------------------


def check_column_references(text: str, profile: DataProfile) -> dict:
    """Check which column names from the profile are mentioned in text.

    Args:
        text:    Artifact text to inspect.
        profile: DataProfile providing the ground-truth column names.

    Returns:
        {
            "present":     [column names that appear in text],
            "unmentioned": [column names that do NOT appear in text],
        }
    """
    text_lower = text.lower()
    present: list[str] = []
    unmentioned: list[str] = []

    for col_name in profile.columns:
        if col_name.lower() in text_lower:
            present.append(col_name)
        else:
            unmentioned.append(col_name)

    return {"present": present, "unmentioned": unmentioned}


# ---------------------------------------------------------------------------
# check_statistic_accuracy
# ---------------------------------------------------------------------------


def check_statistic_accuracy(
    text: str,
    profile: DataProfile,
    tolerance: float = 0.05,
) -> dict:
    """Extract numbers from text and check them against known DataProfile stats.

    Strategy:
        1. Build a flat set of known numeric values from the profile
           (row count, column count, missing percentages, numeric column stats).
        2. Extract every integer and decimal number from the text.
        3. For each extracted number, check if it matches any known value
           within relative tolerance.

    Args:
        text:      Artifact text to inspect.
        profile:   DataProfile providing ground-truth statistics.
        tolerance: Relative tolerance for float comparison (default 5 %).

    Returns:
        {
            "accurate":           [numbers that match at least one known stat],
            "inaccurate":         [numbers that match no known stat],
            "known_values_count": int,
        }
    """
    known: set[float] = set()

    known.add(float(profile.shape[0]))
    known.add(float(profile.shape[1]))

    for pct in profile.missing.values():
        if pct is not None:
            known.add(round(float(pct), 2))

    for cp in profile.columns.values():
        for attr in ("mean", "std", "min", "max", "median", "q25", "q75", "missing_pct"):
            val = getattr(cp, attr, None)
            if val is not None:
                known.add(round(float(val), 2))

    if not known:
        return {"accurate": [], "inaccurate": [], "known_values_count": 0}

    raw_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    extracted = [float(n) for n in raw_numbers]

    accurate: list[float] = []
    inaccurate: list[float] = []

    for num in extracted:
        matched = any(
            num == kv if kv == 0 else abs(num - kv) / abs(kv) <= tolerance for kv in known
        )
        if matched:
            accurate.append(num)
        else:
            inaccurate.append(num)

    return {
        "accurate": accurate,
        "inaccurate": inaccurate,
        "known_values_count": len(known),
    }


# ---------------------------------------------------------------------------
# check_required_sections
# ---------------------------------------------------------------------------


def check_required_sections(text: str, artifact_type: str) -> dict:
    """Check whether required markdown headings are present in text.

    Uses the REQUIRED_SECTIONS mapping from rubrics.py. Heading matching
    is case-insensitive and ignores leading emoji characters.

    Args:
        text:          Markdown artifact to inspect.
        artifact_type: One of "profiler" or "reporter".

    Returns:
        {
            "present": [heading keywords found],
            "missing": [heading keywords NOT found],
        }
    """
    expected = REQUIRED_SECTIONS.get(artifact_type, [])
    text_lower = text.lower()

    present: list[str] = []
    missing: list[str] = []

    for heading in expected:
        if heading.lower() in text_lower:
            present.append(heading)
        else:
            missing.append(heading)

    return {"present": present, "missing": missing}


# ---------------------------------------------------------------------------
# check_insight_fields
# ---------------------------------------------------------------------------


def check_insight_fields(insights: list[dict]) -> dict:
    """Check whether each insight dict contains all required non-empty fields.

    Args:
        insights: List of insight dicts produced by AnalystAgent.

    Returns:
        {
            "valid_count": int,
            "issues": [
                {"index": int, "insight_title": str, "problems": [str]},
                ...
            ],
        }
    """
    issues: list[dict] = []

    for idx, insight in enumerate(insights):
        problems: list[str] = []
        for f in _INSIGHT_REQUIRED_FIELDS:
            if f not in insight:
                problems.append(f"missing field: '{f}'")
            elif not str(insight[f]).strip():
                problems.append(f"empty field: '{f}'")
        if problems:
            issues.append(
                {
                    "index": idx,
                    "insight_title": str(insight.get("title", f"insight_{idx}")),
                    "problems": problems,
                }
            )

    return {
        "valid_count": len(insights) - len(issues),
        "issues": issues,
    }
