"""Rubric definitions and required-section lists for the LLM Judge.

RUBRICS
    Per-artifact list of criteria dicts with name, label, and weight.
    Weights within each artifact sum exactly to 1.0 — verified by tests.

REQUIRED_SECTIONS
    Sections the judge checks for (via validators) before scoring.
    Keys match the artifact_type strings used in EvaluationResult.
"""

from __future__ import annotations

from typing import Any

RUBRICS: dict[str, list[dict[str, Any]]] = {
    "profiler": [
        {"name": "grounding", "label": "Grounding", "weight": 0.35},
        {"name": "completeness", "label": "Completeness", "weight": 0.25},
        {"name": "clarity", "label": "Clarity", "weight": 0.25},
        {"name": "specificity", "label": "Specificity", "weight": 0.15},
    ],
    "analyst": [
        {"name": "factual_correctness", "label": "Factual Correctness", "weight": 0.35},
        {"name": "relevance", "label": "Relevance", "weight": 0.20},
        {"name": "actionability", "label": "Actionability", "weight": 0.20},
        {"name": "priority_calibration", "label": "Priority Calibration", "weight": 0.15},
        {"name": "diversity", "label": "Diversity", "weight": 0.10},
    ],
    "reporter": [
        {"name": "faithfulness", "label": "Faithfulness", "weight": 0.30},
        {"name": "coherence", "label": "Coherence", "weight": 0.25},
        {"name": "language", "label": "Language Quality", "weight": 0.20},
        {"name": "completeness", "label": "Completeness", "weight": 0.15},
        {"name": "actionability", "label": "Actionability", "weight": 0.10},
    ],
}

REQUIRED_SECTIONS: dict[str, list[str]] = {
    "profiler": [
        "Dataset Overview",
        "Data Quality Assessment",
        "Column-by-Column Analysis",
        "Statistical Highlights",
        "Key Takeaways",
    ],
    "reporter": [
        "Executive Summary",
        "Dataset Description",
        "Key Insights",
        "Visualizations",
        "Recommendations",
        "Limitations",
    ],
}
