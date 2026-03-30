"""Rubric definitions for each artifact type.

Each rubric is a list of criteria dicts with keys:
    name   — snake_case identifier (matches LLM response key)
    label  — human-readable label for the UI
    weight — contribution to overall_score (all weights must sum to 1.0)
"""

from __future__ import annotations

RUBRICS: dict[str, list[dict]] = {
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
        {"name": "coherence", "label": "Coherence", "weight": 0.25},
        {"name": "faithfulness", "label": "Faithfulness", "weight": 0.30},
        {"name": "language", "label": "Language Quality", "weight": 0.20},
        {"name": "completeness", "label": "Completeness", "weight": 0.15},
        {"name": "actionability", "label": "Actionability", "weight": 0.10},
    ],
}

# Required markdown section headings per artifact type (used by validators)
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
