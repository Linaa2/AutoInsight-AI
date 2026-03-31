"""LLM client factory for AutoInsight-AI.

Supports Ollama (primary, local) and Gemini (optional cloud fallback).
Model selection is task-aware: use the text model for profiler/analyst/reporter
and the code model for visualizer/text-to-code agents.

All model names and provider settings are read from :mod:`config.settings`
(which in turn reads from ``.env`` / environment variables).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Factory for task-aware LangChain-compatible chat models.

    Usage::

        llm = LLMClient.get_code_llm()    # for visualizer / text-to-code
        llm = LLMClient.get_text_llm()    # for profiler / analyst / reporter
        llm = LLMClient.get_llm("code")   # equivalent to get_code_llm()
    """

    @staticmethod
    def get_text_llm() -> BaseChatModel:
        """Return a chat model for text generation tasks.

        Used by: Profiler, Analyst, Reporter agents.
        Model: ``OLLAMA_TEXT_MODEL`` setting (default ``qwen3:14b``).
        """
        return LLMClient._build(settings.OLLAMA_TEXT_MODEL)

    @staticmethod
    def get_code_llm() -> BaseChatModel:
        """Return a chat model for code generation tasks.

        Used by: Visualizer, Text-to-Code agents.
        Model: ``OLLAMA_CODE_MODEL`` setting (default ``qwen2.5-coder:14b``).
        """
        return LLMClient._build(settings.OLLAMA_CODE_MODEL)

    @staticmethod
    def get_llm(kind: Literal["text", "code"] = "text") -> BaseChatModel:
        """Return a text or code model by task name.

        Args:
            kind: ``"text"`` for generation tasks, ``"code"`` for code tasks.
        """
        if kind == "code":
            return LLMClient.get_code_llm()
        return LLMClient.get_text_llm()

    @staticmethod
    def _build(model: str) -> BaseChatModel:
        """Instantiate the LangChain chat model for the configured provider."""
        if settings.LLM_PROVIDER == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs = LLMClient._with_timeout(
                ChatGoogleGenerativeAI,
                {
                    "model": settings.GEMINI_MODEL,
                    "convert_system_message_to_human": True,
                },
            )
            return ChatGoogleGenerativeAI(
                **kwargs,
            )
        kwargs = LLMClient._with_timeout(
            ChatOllama,
            {
                "model": model,
                "base_url": settings.OLLAMA_BASE_URL,
            },
        )
        return ChatOllama(**kwargs)

    @staticmethod
    def _with_timeout(model_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Attach timeout configuration when the provider supports it.

        LangChain integrations are not fully consistent about the constructor
        parameter name they expose, so we inspect the signature and only pass
        arguments the target class accepts.
        """
        resolved = dict(kwargs)
        try:
            params = inspect.signature(model_cls).parameters
        except (TypeError, ValueError):
            return resolved

        if "timeout" in params:
            resolved.setdefault("timeout", settings.LLM_TIMEOUT)
        elif "request_timeout" in params:
            resolved.setdefault("request_timeout", settings.LLM_TIMEOUT)
        elif "client_kwargs" in params:
            client_kwargs = dict(resolved.get("client_kwargs") or {})
            client_kwargs.setdefault("timeout", settings.LLM_TIMEOUT)
            resolved["client_kwargs"] = client_kwargs

        return resolved


# ---------------------------------------------------------------------------
# Backward-compatible helpers — kept for existing callers
# ---------------------------------------------------------------------------


def call_llm_with_messages(
    system: str,
    human: str,
    model: str | None = None,
    callbacks: list | None = None,
) -> str:
    """Send a system+human message pair and return the model response.

    Legacy helper — prefer ``LLMClient`` + ``ChatPromptTemplate`` for new code.

    Args:
        system:    System prompt content.
        human:     Human turn content.
        model:     Optional model name override.
        callbacks: Optional list of LangChain callbacks (e.g. LangFuse handler).
    """
    llm = LLMClient._build(model or settings.OLLAMA_TEXT_MODEL)
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    try:
        response = llm.invoke(messages, config={"callbacks": callbacks or []})
        return str(response.content)
    except Exception as exc:
        resolved = model or settings.OLLAMA_TEXT_MODEL
        raise RuntimeError(
            f"LLM call failed (model={resolved!r}, provider={settings.LLM_PROVIDER!r}): {exc}"
        ) from exc


def call_llm(prompt: str, model: str | None = None) -> str:
    """Send a single prompt string and return the model response.

    Legacy helper — prefer ``LLMClient`` + ``ChatPromptTemplate`` for new code.
    """
    return call_llm_with_messages(system="", human=prompt, model=model)


def check_ollama_health() -> bool:
    """Return True if the Ollama server is reachable, False otherwise."""
    import urllib.request

    try:
        with urllib.request.urlopen(settings.OLLAMA_BASE_URL, timeout=3):
            return True
    except Exception:
        return False
