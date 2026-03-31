"""Configuration constants for the evaluation subsystem.

All values are read from environment variables with sensible defaults,
so behaviour can be tuned at deployment time without touching code.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Uncertainty Estimator — confidence-level thresholds (integer 0-100)
# ---------------------------------------------------------------------------

EVAL_UNCERTAINTY_HIGH_THRESHOLD: int = int(os.getenv("EVAL_UNCERTAINTY_HIGH_THRESHOLD", "80"))
EVAL_UNCERTAINTY_MEDIUM_THRESHOLD: int = int(os.getenv("EVAL_UNCERTAINTY_MEDIUM_THRESHOLD", "50"))

# ---------------------------------------------------------------------------
# LLM Judge — model and truncation settings
# ---------------------------------------------------------------------------

EVAL_JUDGE_MODEL: str = os.getenv(
    "EVAL_JUDGE_MODEL",
    os.getenv("OLLAMA_TEXT_MODEL", "qwen3:14b"),
)
EVAL_MAX_SECTION_CHARS: int = int(os.getenv("EVAL_MAX_SECTION_CHARS", "300"))
EVAL_MAX_INSIGHT_FIELD_CHARS: int = int(os.getenv("EVAL_MAX_INSIGHT_FIELD_CHARS", "150"))

# ---------------------------------------------------------------------------
# LLM Judge — grade thresholds (float 0.0-1.0)
# ---------------------------------------------------------------------------

EVAL_EXCELLENT_THRESHOLD: float = float(os.getenv("EVAL_EXCELLENT_THRESHOLD", "0.85"))
EVAL_GOOD_THRESHOLD: float = float(os.getenv("EVAL_GOOD_THRESHOLD", "0.70"))
EVAL_FAIR_THRESHOLD: float = float(os.getenv("EVAL_FAIR_THRESHOLD", "0.50"))
