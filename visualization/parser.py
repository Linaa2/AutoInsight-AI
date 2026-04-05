"""Parser and validator for the raw LLM output from the visualizer prompt.

Responsibilities:
    1. Extract a JSON block from raw LLM text (which may include markdown fences
       or prose before/after the JSON).
    2. Validate the top-level structure (``{"charts": [...]}`).
    3. Validate each chart spec against the schema (required fields, allowed types).
    4. Return a clean list of ChartSpec objects and an optional error string.

Design principle: never raise — always return a structured result so the caller
can decide what to do with partial or empty output.
"""

from __future__ import annotations

import json
import re

from visualization.schemas import ALLOWED_CHART_TYPES, ChartSpec

# Required keys that must be present in every chart spec dict.
_REQUIRED_FIELDS: tuple[str, ...] = ("title", "chart_type", "code")


def _extract_json_str(text: str) -> str | None:
    """Pull the first JSON object out of *text*, stripping markdown fences.

    Strategy:
        1. Remove ```json ... ``` or ``` ... ``` fences.
        2. If the response starts with ``[``, it is a bare chart array — wrap it
           in ``{"charts": [...]}`` so the rest of the pipeline sees a uniform shape.
        3. Otherwise, use balanced-bracket matching to extract the first top-level
           ``{...}`` block (avoids the classic first-``{`` / last-``}`` bug that
           produces "Extra data" when multiple JSON objects appear in the text).

    Returns:
        A candidate JSON string, or None if no ``{...}`` block was found.
    """
    # Strip all markdown code fences (```json, ```python, or plain ```)
    text = re.sub(r"```[\w]*\s*", "", text)
    text = re.sub(r"```", "", text).strip()

    # If the LLM returned a bare array, wrap it so the parser sees {"charts": [...]}
    stripped = text.lstrip()
    if stripped.startswith("["):
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start != -1 and arr_end > arr_start:
            return f'{{"charts": {text[arr_start : arr_end + 1]}}}'
        return None

    # Use balanced-bracket matching to find the first complete {...} block.
    # This avoids returning "obj1}, {obj2" when multiple objects live in the text.
    obj_start = text.find("{")
    if obj_start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[obj_start:], start=obj_start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[obj_start : i + 1]

    return None


def _validate_chart_spec(raw: object) -> tuple[ChartSpec | None, str]:
    """Validate a single raw chart spec dict.

    Args:
        raw: The object parsed from JSON (expected to be a dict).

    Returns:
        ``(spec, "")`` on success, or ``(None, error_message)`` on failure.
    """
    if not isinstance(raw, dict):
        return None, f"Chart spec must be a JSON object, got {type(raw).__name__!r}"

    for field in _REQUIRED_FIELDS:
        if field not in raw:
            return None, f"Missing required field {field!r} in chart spec"

    chart_type = str(raw["chart_type"])
    if chart_type not in ALLOWED_CHART_TYPES:
        allowed = ", ".join(sorted(ALLOWED_CHART_TYPES))
        return None, f"Unknown chart_type {chart_type!r}. Allowed: {allowed}"

    # Build the spec with required and optional fields
    spec = ChartSpec(
        title=str(raw["title"]),
        chart_type=chart_type,
        code=str(raw["code"]),
        explanation=str(raw["explanation"]) if "explanation" in raw else None,
        columns_used=(
            [str(c) for c in raw["columns_used"]]
            if "columns_used" in raw and isinstance(raw["columns_used"], list)
            else []
        ),
    )

    return spec, ""


def parse_llm_output(raw: str) -> tuple[list[ChartSpec], str | None]:
    """Extract and validate chart specs from raw LLM output.

    This function is intentionally lenient at the chart level: if some chart
    specs are valid and others are not, the valid ones are returned alongside
    an error message describing the failures.

    Args:
        raw: The raw text string returned by the LLM.

    Returns:
        A tuple ``(charts, error)`` where:
            - ``charts`` is a (possibly empty) list of validated ChartSpec dicts.
            - ``error`` is None on full success, or a string describing what
              went wrong (JSON parse error, missing fields, bad chart types, …).
    """
    # --- Step 1: extract a JSON candidate from the raw text ---
    json_candidate = _extract_json_str(raw)
    if json_candidate is None:
        return [], "Could not find a JSON object in the LLM output."

    # --- Step 2: parse JSON ---
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        return [], f"JSON parse error: {exc}"

    # --- Step 3: validate top-level structure ---
    if not isinstance(parsed, dict):
        return [], "Expected a JSON object at the top level."
    if "charts" not in parsed:
        return [], "JSON object is missing the required 'charts' key."
    if not isinstance(parsed["charts"], list):
        return [], "'charts' must be a JSON array."

    # --- Step 4: validate each chart spec ---
    valid_specs: list[ChartSpec] = []
    errors: list[str] = []

    for i, item in enumerate(parsed["charts"]):
        spec, err = _validate_chart_spec(item)
        if spec is not None:
            valid_specs.append(spec)
        else:
            errors.append(f"Chart #{i + 1}: {err}")

    # Build a combined error message if any specs failed validation
    combined_error: str | None = None
    if errors:
        combined_error = "; ".join(errors)

    return valid_specs, combined_error
