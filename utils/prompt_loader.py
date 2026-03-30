"""Shared prompt-loading utility.

Every agent that needs prompts from ``config/prompts.yaml`` should use::

    from utils.prompt_loader import load_prompt_section

    prompts = load_prompt_section("profiler")   # {"system": "...", "human": "..."}
    prompts = load_prompt_section("visualizer")

The YAML path comes from :mod:`config.settings` — agents never compute it.
"""

from __future__ import annotations

from typing import Any

import yaml

from config.settings import PROMPTS_PATH


def load_prompt_section(section: str) -> dict[str, Any]:
    """Load a named section from the shared prompts YAML file.

    Args:
        section: Top-level key in ``config/prompts.yaml``
                 (e.g. ``"profiler"``, ``"visualizer"``).

    Returns:
        The dictionary stored under *section* (typically contains
        ``"system"`` and ``"human"`` keys).

    Raises:
        KeyError: If *section* does not exist in the prompts file.
        FileNotFoundError: If the prompts file does not exist.
    """
    with PROMPTS_PATH.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)

    if section not in data:
        raise KeyError(f"Section {section!r} not found in {PROMPTS_PATH}")

    return dict(data[section])
