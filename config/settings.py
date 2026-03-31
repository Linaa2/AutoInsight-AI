"""Centralized project configuration for AutoInsight-AI.

This module is the **single source of truth** for:

- **Static paths** — repository root, prompts file, docs directory.
  These are deterministic filesystem locations that do **not** belong
  in ``.env``.
- **Runtime settings** — env-driven values (LLM provider, model names,
  timeout, profiler knobs) that vary per developer or deployment.

Usage::

    from config.settings import settings

    prompts = settings.PROMPTS_PATH   # Path to config/prompts.yaml
    model   = settings.OLLAMA_TEXT_MODEL
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Repo layout — computed once, never from env
# ---------------------------------------------------------------------------

#: Project root (directory containing ``pyproject.toml``).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Path to the shared YAML prompt file.
PROMPTS_PATH: Path = REPO_ROOT / "config" / "prompts.yaml"

#: Path to the ``docs/`` directory.
DOCS_DIR: Path = REPO_ROOT / "docs"

#: Path to ``data/`` for sample datasets.
DATA_DIR: Path = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Runtime settings (read from environment / .env)
# ---------------------------------------------------------------------------


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration.

    Values are read from environment variables (or ``.env``).
    Frozen so they cannot be accidentally mutated at runtime.
    """

    # -- LLM --
    LLM_PROVIDER: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    OLLAMA_BASE_URL: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    OLLAMA_TEXT_MODEL: str = field(default_factory=lambda: _env("OLLAMA_TEXT_MODEL", "qwen3:14b"))
    OLLAMA_CODE_MODEL: str = field(
        default_factory=lambda: _env("OLLAMA_CODE_MODEL", "qwen2.5-coder:14b")
    )
    LLM_TIMEOUT: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 60))
    GEMINI_MODEL: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-1.5-flash"))

    # -- Profiler --
    PROFILER_SAMPLE_ROWS: int = field(default_factory=lambda: _env_int("PROFILER_SAMPLE_ROWS", 5))
    PROFILER_TOP_VALUES: int = field(default_factory=lambda: _env_int("PROFILER_TOP_VALUES", 10))

    # -- Embeddings / ChromaDB --
    OLLAMA_EMBED_MODEL: str = field(
        default_factory=lambda: _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    CHROMA_DIR: str = field(default_factory=lambda: _env("CHROMA_DIR", "./chroma_db"))

    # -- Data loader --
    DATA_LOADER_EXCEL_SHEET: str = field(
        default_factory=lambda: _env("DATA_LOADER_EXCEL_SHEET", "0")
    )

    # -- App --
    APP_TITLE: str = field(default_factory=lambda: _env("APP_TITLE", "AutoInsight AI"))

    # -- LangFuse external observability (optional) --
    LANGFUSE_ENABLED: bool = field(default_factory=lambda: _env_bool("LANGFUSE_ENABLED", False))
    LANGFUSE_HOST: str = field(
        default_factory=lambda: _env("LANGFUSE_HOST", "http://localhost:3001")
    )
    LANGFUSE_PUBLIC_KEY: str = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY", ""))
    LANGFUSE_SECRET_KEY: str = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY", ""))
    LANGFUSE_ENV: str = field(default_factory=lambda: _env("LANGFUSE_ENV", "development"))
    LANGFUSE_RELEASE: str = field(default_factory=lambda: _env("LANGFUSE_RELEASE", ""))


#: Module-level singleton — import ``settings`` everywhere.
settings = Settings()
