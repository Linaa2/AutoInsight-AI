"""Evaluation system configuration — all settings backed by environment variables.

Every constant here can be overridden via the environment without touching code.
Add the corresponding keys to your .env file to customise behaviour.

Environment variables::

    EVAL_JUDGE_MODEL            = qwen3:14b
    EVAL_MAX_SECTION_CHARS      = 300
    EVAL_MAX_INSIGHT_FIELD_CHARS= 150
    EVAL_EXCELLENT_THRESHOLD    = 0.85
    EVAL_GOOD_THRESHOLD         = 0.70
    EVAL_FAIR_THRESHOLD         = 0.50
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------

EVAL_JUDGE_MODEL: str = os.getenv(
    "EVAL_JUDGE_MODEL",
    os.getenv("OLLAMA_TEXT_MODEL", "qwen3:14b"),
)
"""LLM model used by EvaluationAgent.
Defaults to OLLAMA_TEXT_MODEL (qwen3:14b).
qwen3:14b is preferred over the coder model for text-reasoning evaluation tasks.
"""

# ---------------------------------------------------------------------------
# Truncation limits
# ---------------------------------------------------------------------------

EVAL_MAX_SECTION_CHARS: int = int(os.getenv("EVAL_MAX_SECTION_CHARS", "300"))
"""Max characters kept per markdown section body when truncating for the judge prompt."""

EVAL_MAX_INSIGHT_FIELD_CHARS: int = int(os.getenv("EVAL_MAX_INSIGHT_FIELD_CHARS", "150"))
"""Max characters kept per text field inside each insight dict."""

# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------

EVAL_EXCELLENT_THRESHOLD: float = float(os.getenv("EVAL_EXCELLENT_THRESHOLD", "0.85"))
"""Minimum overall_score to receive grade "excellent"."""

EVAL_GOOD_THRESHOLD: float = float(os.getenv("EVAL_GOOD_THRESHOLD", "0.70"))
"""Minimum overall_score to receive grade "good"."""

EVAL_FAIR_THRESHOLD: float = float(os.getenv("EVAL_FAIR_THRESHOLD", "0.50"))
"""Minimum overall_score to receive grade "fair". Below this is "poor"."""
