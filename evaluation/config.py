"""Configuration constants for the evaluation subsystem.

All values are read from environment variables with sensible defaults,
so behaviour can be tuned at deployment time without touching code.
"""

from __future__ import annotations

import os

from config.settings import settings

# ---------------------------------------------------------------------------
# Uncertainty Estimator — confidence-level thresholds (integer 0-100)
# ---------------------------------------------------------------------------

EVAL_UNCERTAINTY_HIGH_THRESHOLD: int = int(os.getenv("EVAL_UNCERTAINTY_HIGH_THRESHOLD", "80"))
EVAL_UNCERTAINTY_MEDIUM_THRESHOLD: int = int(os.getenv("EVAL_UNCERTAINTY_MEDIUM_THRESHOLD", "50"))
EVAL_UNCERTAINTY_BATCH_SIZE: int = int(os.getenv("EVAL_UNCERTAINTY_BATCH_SIZE", "6"))

# ---------------------------------------------------------------------------
# LLM Judge — model and truncation settings
# ---------------------------------------------------------------------------

_eval_model_override = os.getenv("EVAL_JUDGE_MODEL", "").strip()
if _eval_model_override:
    EVAL_JUDGE_MODEL: str = _eval_model_override
elif settings.LLM_PROVIDER == "gemini":
    EVAL_JUDGE_MODEL = settings.GEMINI_MODEL
elif settings.LLM_PROVIDER == "groq":
    EVAL_JUDGE_MODEL = settings.GROQ_MODEL
elif settings.LLM_PROVIDER == "openrouter":
    EVAL_JUDGE_MODEL = settings.OPENROUTER_MODEL
else:
    EVAL_JUDGE_MODEL = settings.OLLAMA_TEXT_MODEL

# Judge calls are offline evaluation work rather than user-facing pipeline steps,
# so give them a dedicated budget and never default below 300 seconds.
EVAL_JUDGE_TIMEOUT: int = int(os.getenv("EVAL_JUDGE_TIMEOUT", str(max(300, settings.LLM_TIMEOUT))))
EVAL_MAX_SECTION_CHARS: int = int(os.getenv("EVAL_MAX_SECTION_CHARS", "300"))
EVAL_MAX_INSIGHT_FIELD_CHARS: int = int(os.getenv("EVAL_MAX_INSIGHT_FIELD_CHARS", "150"))

# ---------------------------------------------------------------------------
# LLM Judge — grade thresholds (float 0.0-1.0)
# ---------------------------------------------------------------------------

EVAL_EXCELLENT_THRESHOLD: float = float(os.getenv("EVAL_EXCELLENT_THRESHOLD", "0.85"))
EVAL_GOOD_THRESHOLD: float = float(os.getenv("EVAL_GOOD_THRESHOLD", "0.70"))
EVAL_FAIR_THRESHOLD: float = float(os.getenv("EVAL_FAIR_THRESHOLD", "0.50"))
