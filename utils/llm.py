"""LLM client factory supporting Ollama (local) and Google Gemini."""

import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


class LLMClient:
    """Creates and returns a LangChain-compatible chat model.

    Environment variables:
        LLM_PROVIDER:      "ollama" (default) or "gemini"
        OLLAMA_MODEL:      Ollama model name, e.g. "mistral"
        OLLAMA_BASE_URL:   Ollama server URL, e.g. "http://localhost:11434"
        GEMINI_MODEL:      Gemini model name, e.g. "gemini-1.5-flash"
        GOOGLE_API_KEY:    Required when LLM_PROVIDER=gemini
        LLM_TIMEOUT:       Request timeout in seconds (default 60)
    """

    def __init__(self) -> None:
        self._provider = os.getenv("LLM_PROVIDER", "ollama")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._timeout = int(os.getenv("LLM_TIMEOUT", "60"))

    def get_llm(self) -> BaseChatModel:
        """Return the configured LLM instance."""
        if self._provider == "gemini":
            return self._build_gemini()
        return self._build_ollama()

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_ollama(self) -> BaseChatModel:
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            model=self._ollama_model,
            base_url=self._ollama_base_url,
            timeout=self._timeout,
        )

    def _build_gemini(self) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is required when LLM_PROVIDER=gemini"
            )
        return ChatGoogleGenerativeAI(
            model=self._gemini_model,
            google_api_key=api_key,
            timeout=self._timeout,
        )
