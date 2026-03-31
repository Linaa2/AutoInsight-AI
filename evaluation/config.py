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
