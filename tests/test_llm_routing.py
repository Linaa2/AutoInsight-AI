"""Tests for centralized model routing in ``utils.llm``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _mock_settings() -> SimpleNamespace:
    return SimpleNamespace(
        LLM_PROVIDER="ollama",
        OLLAMA_LIGHT_MODEL="qwen3:4b",
        OLLAMA_TEXT_MODEL="qwen3:14b",
        OLLAMA_CODE_MODEL="qwen2.5-coder:14b",
        OLLAMA_BASE_URL="http://localhost:11434",
        GEMINI_MODEL="gemini-1.5-flash",
        LLM_TIMEOUT=300,
    )


def test_default_model_uses_light_tier_for_light_tasks() -> None:
    with patch("utils.llm.settings", _mock_settings()):
        from utils.llm import LLMClient

        assert LLMClient._default_model("light") == "qwen3:4b"


def test_get_task_llm_routes_profiler_to_light_model() -> None:
    with (
        patch("utils.llm.settings", _mock_settings()),
        patch("utils.llm.LLMClient._build", return_value=MagicMock()) as mock_build,
    ):
        from utils.llm import LLMClient

        LLMClient.get_task_llm("profiler")

    mock_build.assert_called_once_with("qwen3:4b")


def test_get_task_llm_routes_reporter_to_standard_text_model() -> None:
    with (
        patch("utils.llm.settings", _mock_settings()),
        patch("utils.llm.LLMClient._build", return_value=MagicMock()) as mock_build,
    ):
        from utils.llm import LLMClient

        LLMClient.get_task_llm("reporter")

    mock_build.assert_called_once_with("qwen3:14b")


def test_call_llm_with_messages_uses_task_routing_when_model_is_omitted() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="ok")

    with (
        patch("utils.llm.settings", _mock_settings()),
        patch("utils.llm.LLMClient._build", return_value=mock_llm) as mock_build,
    ):
        from utils.llm import call_llm_with_messages

        result = call_llm_with_messages("system", "human", task="uncertainty")

    assert result == "ok"
    mock_build.assert_called_once_with("qwen3:4b", timeout=None)


def test_call_llm_with_messages_explicit_model_overrides_task_routing() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="ok")

    with (
        patch("utils.llm.settings", _mock_settings()),
        patch("utils.llm.LLMClient._build", return_value=mock_llm) as mock_build,
    ):
        from utils.llm import call_llm_with_messages

        call_llm_with_messages("system", "human", model="custom-model", task="profiler")

    mock_build.assert_called_once_with("custom-model", timeout=None)
