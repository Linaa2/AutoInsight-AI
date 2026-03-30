"""LLM client factory for AutoInsight-AI.

Supports Ollama (primary, local) and Gemini (optional cloud fallback).
Model selection is task-aware: use the text model for profiler/analyst/reporter
and the code model for visualizer/text-to-code agents.

All model names and provider settings are read from :mod:`config.settings`
(which in turn reads from ``.env`` / environment variables).

Apple Silicon note: Ollama manages Metal (MPS) acceleration internally.
Do NOT pass device="mps" or similar flags to ChatOllama — LangChain's
ChatOllama client does not expose a device parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

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

    Apple Silicon note:
        Ollama handles Metal acceleration via the Ollama runtime daemon.
        No ``device`` parameter is passed to ``ChatOllama`` — it does not
        support one.  Configure GPU layers in Ollama settings if needed.
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
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                convert_system_message_to_human=True,
            )
        return ChatOllama(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
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
    """
    llm = LLMClient._build(model or settings.OLLAMA_TEXT_MODEL)
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    try:
        response = llm.invoke(messages)
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
