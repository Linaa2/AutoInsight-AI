"""LLM client factory for AutoInsight-AI.

Supports Ollama (primary, local) and Gemini (optional cloud fallback).
For Ollama, model routing is task-aware:

- ``OLLAMA_LIGHT_MODEL`` for short, low-complexity text tasks
- ``OLLAMA_TEXT_MODEL`` for standard reasoning / synthesis
- ``OLLAMA_CODE_MODEL`` for code generation

All model names and provider settings are read from :mod:`config.settings`
(which in turn reads from ``.env`` / environment variables).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from config.settings import settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

LLMKind = Literal["light", "text", "code"]
LLMTask = Literal[
    "profiler",
    "analyst",
    "critic",
    "reporter",
    "uncertainty",
    "categorizer",
    "visualizer",
]

_TASK_TO_KIND: dict[LLMTask, LLMKind] = {
    "profiler": "light",
    "analyst": "text",
    "critic": "text",
    "reporter": "text",
    "uncertainty": "light",
    "categorizer": "light",
    "visualizer": "code",
}


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Factory for task-aware LangChain-compatible chat models.

    Usage::

        llm = LLMClient.get_light_llm()   # for fast, low-complexity text tasks
        llm = LLMClient.get_code_llm()    # for visualizer / text-to-code
        llm = LLMClient.get_text_llm()    # for standard reasoning / synthesis
        llm = LLMClient.get_llm("code")   # equivalent to get_code_llm()
        llm = LLMClient.get_task_llm("profiler")
    """

    @staticmethod
    def get_light_llm() -> BaseChatModel:
        """Return a chat model for short, low-complexity text tasks."""
        return LLMClient._build(LLMClient._default_model("light"), _kind="light")

    @staticmethod
    def get_text_llm() -> BaseChatModel:
        """Return a chat model for text generation tasks.

        Used by: Analyst, Critic, Reporter agents.
        Model: ``OLLAMA_TEXT_MODEL`` setting (default ``qwen3:14b``).
        """
        return LLMClient._build(LLMClient._default_model("text"), _kind="text")

    @staticmethod
    def get_code_llm() -> BaseChatModel:
        """Return a chat model for code generation tasks.

        Used by: Visualizer, Text-to-Code agents.
        Model: ``OLLAMA_CODE_MODEL`` setting (default ``qwen2.5-coder:14b``).
        """
        return LLMClient._build(LLMClient._default_model("code"), _kind="code")

    @staticmethod
    def get_llm(kind: LLMKind = "text") -> BaseChatModel:
        """Return a light, text, or code model by complexity tier.

        Args:
            kind: ``"light"`` | ``"text"`` | ``"code"``.
        """
        if kind == "light":
            return LLMClient.get_light_llm()
        if kind == "code":
            return LLMClient.get_code_llm()
        return LLMClient.get_text_llm()

    @staticmethod
    def get_task_llm(task: LLMTask) -> BaseChatModel:
        """Return the routed model for a concrete pipeline task."""
        return LLMClient.get_llm(LLMClient._task_kind(task))

    @staticmethod
    def _task_kind(task: LLMTask) -> LLMKind:
        """Map a task name to its complexity tier."""
        return _TASK_TO_KIND[task]

    @staticmethod
    def _default_model(kind: LLMKind = "text") -> str:
        """Return the provider-appropriate default model for *kind*."""
        if settings.LLM_PROVIDER == "gemini":
            return settings.GEMINI_MODEL
        if settings.LLM_PROVIDER == "groq":
            return settings.GROQ_MODEL
        if settings.LLM_PROVIDER == "openrouter":
            return settings.OPENROUTER_MODEL
        if kind == "light":
            return settings.OLLAMA_LIGHT_MODEL
        if kind == "code":
            return settings.OLLAMA_CODE_MODEL
        return settings.OLLAMA_TEXT_MODEL

    @staticmethod
    def _build(
        model: str,
        *,
        timeout: int | None = None,
        keep_alive: int | str | None = None,
        _kind: LLMKind = "text",
    ) -> BaseChatModel:
        """Instantiate the LangChain chat model for the configured provider.

        When provider is groq or gemini, the model is wrapped with
        ``.with_fallbacks([ollama])`` so rate-limit / network errors
        transparently retry on Ollama.
        """
        if settings.LLM_PROVIDER == "openrouter":
            from langchain_openai import ChatOpenAI

            _openai_llm = ChatOpenAI(
                model=model,
                api_key=SecretStr(settings.OPENROUTER_API_KEY),
                base_url="https://openrouter.ai/api/v1",
                temperature=0,
                max_tokens=4096,  # type: ignore[call-arg]
            )
            fallback = LLMClient._build_ollama(
                LLMClient._ollama_model_for_kind(_kind),
                timeout=timeout,
                keep_alive=keep_alive,
            )
            return cast("BaseChatModel", _openai_llm.with_fallbacks([fallback]))

        if settings.LLM_PROVIDER == "groq":
            from langchain_groq import ChatGroq

            _groq_llm = ChatGroq(
                model=model,
                api_key=SecretStr(settings.GROQ_API_KEY),
                temperature=0,
                max_tokens=4096,
            )
            fallback = LLMClient._build_ollama(
                LLMClient._ollama_model_for_kind(_kind),
                timeout=timeout,
                keep_alive=keep_alive,
            )
            return cast("BaseChatModel", _groq_llm.with_fallbacks([fallback]))

        if settings.LLM_PROVIDER == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs = LLMClient._with_timeout(
                ChatGoogleGenerativeAI,
                {
                    "model": model,
                    "convert_system_message_to_human": True,
                },
                timeout=timeout,
            )
            _gemini_llm = ChatGoogleGenerativeAI(**kwargs)
            fallback = LLMClient._build_ollama(
                LLMClient._ollama_model_for_kind(_kind),
                timeout=timeout,
                keep_alive=keep_alive,
            )
            return cast("BaseChatModel", _gemini_llm.with_fallbacks([fallback]))

        kwargs = LLMClient._with_timeout(
            ChatOllama,
            {
                "model": model,
                "base_url": settings.OLLAMA_BASE_URL,
            },
            timeout=timeout,
            keep_alive=keep_alive,
        )
        return ChatOllama(**kwargs)

    @staticmethod
    def _with_timeout(
        model_cls: type,
        kwargs: dict[str, Any],
        *,
        timeout: int | None = None,
        keep_alive: int | str | None = None,
    ) -> dict[str, Any]:
        """Attach timeout configuration when the provider supports it.

        LangChain integrations are not fully consistent about the constructor
        parameter name they expose, so we inspect the signature and only pass
        arguments the target class accepts.
        """
        resolved = dict(kwargs)
        timeout_value = settings.LLM_TIMEOUT if timeout is None else timeout
        try:
            params = inspect.signature(model_cls).parameters
        except (TypeError, ValueError):
            return resolved

        if "timeout" in params:
            resolved.setdefault("timeout", timeout_value)
        elif "request_timeout" in params:
            resolved.setdefault("request_timeout", timeout_value)
        else:
            # ChatOllama (langchain_ollama) routes sync calls through a sync
            # httpx client configured via `sync_client_kwargs`, and async calls
            # through one configured via `client_kwargs` / `async_client_kwargs`.
            # We must set the timeout on BOTH; otherwise synchronous chain.invoke()
            # calls fall back to httpx's 5-second read-timeout default, which is
            # far too short for large models that need to load from disk.
            for ck_key in ("client_kwargs", "async_client_kwargs", "sync_client_kwargs"):
                if ck_key in params:
                    ck = dict(resolved.get(ck_key) or {})
                    ck.setdefault("timeout", timeout_value)
                    resolved[ck_key] = ck

        # Keep the model loaded in Ollama between pipeline steps so subsequent
        # nodes don't pay a cold-start penalty (model load can take 30-60 s).
        if "keep_alive" in params:
            keep_alive_value = timeout_value if keep_alive is None else keep_alive
            resolved.setdefault("keep_alive", keep_alive_value)

        return resolved

    @staticmethod
    def _build_ollama(
        model: str,
        *,
        timeout: int | None = None,
        keep_alive: int | str | None = None,
    ) -> BaseChatModel:
        """Build an Ollama model instance (used as fallback)."""
        kwargs = LLMClient._with_timeout(
            ChatOllama,
            {"model": model, "base_url": settings.OLLAMA_BASE_URL},
            timeout=timeout,
            keep_alive=keep_alive,
        )
        return ChatOllama(**kwargs)

    @staticmethod
    def _ollama_model_for_kind(kind: LLMKind = "text") -> str:
        """Return the Ollama model name for a given complexity tier."""
        if kind == "light":
            return settings.OLLAMA_LIGHT_MODEL
        if kind == "code":
            return settings.OLLAMA_CODE_MODEL
        return settings.OLLAMA_TEXT_MODEL


# ---------------------------------------------------------------------------
# Backward-compatible helpers — kept for existing callers
# ---------------------------------------------------------------------------


def call_llm_with_messages(
    system: str,
    human: str,
    model: str | None = None,
    callbacks: list | None = None,
    timeout: int | None = None,
    *,
    kind: LLMKind = "text",
    task: LLMTask | None = None,
) -> str:
    """Send a system+human message pair and return the model response.

    Legacy helper — prefer ``LLMClient`` + ``ChatPromptTemplate`` for new code.

    Args:
        system:    System prompt content.
        human:     Human turn content.
        model:     Optional model name override.
        callbacks: Optional list of LangChain callbacks (e.g. LangFuse handler).
        timeout:   Optional per-call timeout override in seconds.
        kind:      Complexity tier used when *model* is omitted.
        task:      Concrete task name used to route to the right complexity tier.
    """
    resolved_kind = LLMClient._task_kind(task) if task is not None else kind
    resolved_model = model or LLMClient._default_model(resolved_kind)
    llm = LLMClient._build(resolved_model, timeout=timeout, _kind=resolved_kind)
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    try:
        response = llm.invoke(messages, config={"callbacks": callbacks or []})
        content = response.content
        # Gemini models may return a list of content blocks instead of a plain string
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)
    except Exception as exc:
        raise RuntimeError(
            f"LLM call failed (model={resolved_model!r}, provider={settings.LLM_PROVIDER!r}): {exc}"
        ) from exc


def call_llm(
    prompt: str,
    model: str | None = None,
    *,
    kind: LLMKind = "text",
    task: LLMTask | None = None,
) -> str:
    """Send a single prompt string and return the model response.

    Legacy helper — prefer ``LLMClient`` + ``ChatPromptTemplate`` for new code.
    """
    return call_llm_with_messages(system="", human=prompt, model=model, kind=kind, task=task)


def check_ollama_health() -> bool:
    """Return True if the Ollama server is reachable, False otherwise."""
    import urllib.request

    try:
        with urllib.request.urlopen(settings.OLLAMA_BASE_URL, timeout=3):
            return True
    except Exception:
        return False
