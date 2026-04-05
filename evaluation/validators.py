"""Deterministic validators for the LLM-as-Judge evaluation subsystem.

Each validator is called before the corresponding LLM judge call.
Its output is injected into the judge prompt as ground truth, so the LLM
confirms or nuances rather than re-discovering facts from scratch.

Functions
---------
check_required_sections(text, artifact_type)  -> dict
check_column_references(text, profile_data)   -> dict
check_statistic_accuracy(text, profile_data)  -> dict
check_insight_fields(insights)                -> dict
"""

from __future__ import annotations

from typing import Any

from evaluation.rubrics import REQUIRED_SECTIONS


def check_required_sections(text: str, artifact_type: str) -> dict[str, Any]:
    """Check which required markdown sections are present or missing.

    Args:
        text:          The artifact markdown text.
        artifact_type: One of ``"profiler"`` or ``"reporter"`` (analyst has no
                       fixed section list).

    Returns:
        {
            "required":  list of section names that should be present,
            "present":   list of section names found in *text* (case-insensitive),
            "missing":   list of section names not found in *text*,
        }
    """
    required: list[str] = REQUIRED_SECTIONS.get(artifact_type, [])
    present: list[str] = []
    missing: list[str] = []
    text_lower = text.lower()

    for section in required:
        if section.lower() in text_lower:
            present.append(section)
        else:
            missing.append(section)

    return {"required": required, "present": present, "missing": missing}


def check_column_references(text: str, profile_data: dict[str, Any]) -> dict[str, Any]:
    """Check which dataset columns are referenced or absent in the text.

    Args:
        text:         The artifact text to scan.
        profile_data: Pipeline ``profile_data`` dict (must have ``"columns"`` key).

    Returns:
        {
            "total_columns": int,
            "present":       list of column names found in *text*,
            "unmentioned":   list of column names not found in *text*,
        }
    """
    columns: list[str] = list(profile_data.get("columns", {}).keys())
    text_lower = text.lower()
    present: list[str] = []
    unmentioned: list[str] = []

    for col in columns:
        if col.lower() in text_lower:
            present.append(col)
        else:
            unmentioned.append(col)

    return {
        "total_columns": len(columns),
        "present": present,
        "unmentioned": unmentioned,
    }


def check_statistic_accuracy(text: str, profile_data: dict[str, Any]) -> dict[str, Any]:
    """Extract key ground-truth statistics and check whether they appear in text.

    A statistic is considered *verified* when its rounded numeric value (or
    the value with a ``%`` suffix) appears as a substring in the text.

    Args:
        text:         The artifact text to scan.
        profile_data: Pipeline ``profile_data`` dict.

    Returns:
        {
            "ground_truth": list of {"stat": str, "value": str},
            "verified":     list of stat names found in *text*,
            "unverified":   list of stat names not clearly referenced in *text*,
        }
    """
    shape: list[int] = profile_data.get("shape", [0, 0])
    stats: dict[str, Any] = profile_data.get("stats", {})

    ground_truth: list[dict[str, str]] = [
        {"stat": "row_count", "value": str(shape[0])},
        {"stat": "column_count", "value": str(shape[1])},
        {
            "stat": "total_missing_pct",
            "value": f"{float(stats.get('total_missing_pct', 0.0)):.1f}%",
        },
        {
            "stat": "duplicates_pct",
            "value": f"{float(stats.get('duplicates_pct', 0.0)):.1f}%",
        },
    ]

    verified: list[str] = []
    unverified: list[str] = []

    for item in ground_truth:
        raw_value = item["value"].rstrip("%")
        try:
            num = float(raw_value)
            # Accept the number in various plausible formats
            variants = [
                str(int(num)),
                f"{num:.1f}",
                f"{int(num)}%",
                f"{num:.1f}%",
            ]
            if any(v in text for v in variants):
                verified.append(item["stat"])
            else:
                unverified.append(item["stat"])
        except ValueError:
            if raw_value in text:
                verified.append(item["stat"])
            else:
                unverified.append(item["stat"])

    return {
        "ground_truth": ground_truth,
        "verified": verified,
        "unverified": unverified,
    }


def check_insight_fields(insights: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate that each insight dict contains all required fields.

    Required fields: ``title``, ``observation``, ``hypothesis``,
    ``recommendation``, ``priority``.

    Args:
        insights: List of insight dicts from the analyst agent.

    Returns:
        {
            "total":       int — total number of insights,
            "valid_count": int — insights with all required fields,
            "issues":      list[str] — one entry per insight with missing fields,
        }
    """
    required_fields: set[str] = {
        "title",
        "observation",
        "hypothesis",
        "recommendation",
        "priority",
    }
    issues: list[str] = []
    valid_count: int = 0

    for i, ins in enumerate(insights, 1):
        missing = required_fields - set(ins.keys())
        if missing:
            issues.append(f"Insight {i}: missing fields {sorted(missing)}")
        else:
            valid_count += 1

    return {
        "total": len(insights),
        "valid_count": valid_count,
        "issues": issues,
    }
