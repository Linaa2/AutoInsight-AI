"""LLM client factory for AutoInsight-AI.

Supports Ollama (primary, local) and Gemini (optional cloud fallback).
Model selection is task-aware: use the text model for profiler/analyst/reporter
and the code model for visualizer/text-to-code agents.

Configuration (environment variables)::

    LLM_PROVIDER       = ollama          # or gemini
    OLLAMA_BASE_URL    = http://localhost:11434
    OLLAMA_TEXT_MODEL  = qwen3:14b
    OLLAMA_CODE_MODEL  = qwen2.5-coder:14b
    LLM_TIMEOUT        = 60

Apple Silicon note: Ollama manages Metal (MPS) acceleration internally.
Do NOT pass device="mps" or similar flags to ChatOllama — LangChain's
ChatOllama client does not expose a device parameter.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# ---------------------------------------------------------------------------
# Runtime configuration — read once from environment
# ---------------------------------------------------------------------------

_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
_HOST: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_TEXT_MODEL: str = os.getenv("OLLAMA_TEXT_MODEL", "qwen3:14b")
_CODE_MODEL: str = os.getenv("OLLAMA_CODE_MODEL", "qwen2.5-coder:14b")
_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))
_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Factory for task-aware LangChain-compatible chat models.

    Usage::

        llm = LLMClient.get_code_llm()    # for visualizer / text-to-code
        llm = LLMClient.get_text_llm()    # for profiler / analyst / reporter
        llm = LLMClient.get_llm("code")   # equivalent to get_code_llm()

    Apple Silicon note:
        Ollama handles Metal acceleration via the Ollama runtime daemon.
        No ``device`` parameter is passed to ``ChatOllama`` — it does not
        support one.  Configure GPU layers in Ollama settings if needed.
    """

    @staticmethod
    def get_text_llm() -> BaseChatModel:
        """Return a chat model for text generation tasks.

        Used by: Profiler, Analyst, Reporter agents.
        Default model: ``OLLAMA_TEXT_MODEL`` env var (qwen3:14b).
        """
        return LLMClient._build(_TEXT_MODEL)

    @staticmethod
    def get_code_llm() -> BaseChatModel:
        """Return a chat model for code generation tasks.

        Used by: Visualizer, Text-to-Code agents.
        Default model: ``OLLAMA_CODE_MODEL`` env var (qwen2.5-coder:14b).
        """
        return LLMClient._build(_CODE_MODEL)

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
        if _PROVIDER == "gemini":
            # Gemini API does not natively support system messages;
            # convert_system_message_to_human merges them into the human turn.
            return ChatGoogleGenerativeAI(
                model=_GEMINI_MODEL,
                convert_system_message_to_human=True,
            )
        # Ollama (default) — no device parameter; Ollama handles acceleration.
        return ChatOllama(
            model=model,
            base_url=_HOST,
            timeout=_TIMEOUT,
        )


# ---------------------------------------------------------------------------
# Backward-compatible helpers — kept for existing callers
# ---------------------------------------------------------------------------


def call_llm_with_messages(
    system: str,
    human: str,
    model: str | None = None,
) -> str:
    """Send a system+human message pair and return the model response.

    Legacy helper — prefer ``LLMClient`` + ``ChatPromptTemplate`` for new code.

    Args:
        system: The system-role prompt.
        human:  The user-context prompt.
        model:  Optional Ollama model override (ignored when provider is Gemini).

    Raises:
        RuntimeError: On any LLM / network failure.
    """
    llm = LLMClient._build(model or _TEXT_MODEL)
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    try:
        response = llm.invoke(messages)
        return str(response.content)
    except Exception as exc:
        resolved = model or _TEXT_MODEL
        raise RuntimeError(
            f"LLM call failed (model={resolved!r}, provider={_PROVIDER!r}): {exc}"
        ) from exc


def call_llm(prompt: str, model: str | None = None) -> str:
    """Send a single prompt string and return the model response.

    Legacy helper — prefer ``LLMClient`` + ``ChatPromptTemplate`` for new code.
    """
    return call_llm_with_messages(system="", human=prompt, model=model)
