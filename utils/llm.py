"""LLM client factory supporting Ollama (local) and Google Gemini + monitoring utilities."""

import logging
import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────── Default config ───────────────────────────────

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


# ──────────────────────────── LLMClient (shared) ───────────────────────────


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
        self._ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        self._gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._timeout = int(os.getenv("LLM_TIMEOUT", "60"))

    def get_llm(self) -> BaseChatModel:
        """Return the configured LLM instance."""
        if self._provider == "gemini":
            return self._build_gemini()
        return self._build_ollama()

    def _build_ollama(self) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=self._ollama_model,
            base_url=self._ollama_base_url,
            timeout=self._timeout,
        )

    def _build_gemini(self) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise OSError(
                "GOOGLE_API_KEY environment variable is required when LLM_PROVIDER=gemini"
            )
        return ChatGoogleGenerativeAI(
            model=self._gemini_model,
            google_api_key=api_key,
            timeout=self._timeout,
        )


# ──────────────────────────── Functional API (used by analyst) ─────────────


def get_llm(
    temperature: float = 0.3,
    model: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Return an LLM instance with per-call overrides for temperature, model, base_url."""
    provider = os.getenv("LLM_PROVIDER", "ollama")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise OSError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=api_key,
            temperature=temperature,
        )

    from langchain_ollama import ChatOllama

    resolved_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    resolved_model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    logger.info(f"LLM: {resolved_model} @ {resolved_url} (temp={temperature})")

    return ChatOllama(
        base_url=resolved_url,
        model=resolved_model,
        temperature=temperature,
    )


def get_embeddings(
    model: str | None = None,
    base_url: str | None = None,
):
    """Return the Ollama embeddings model (for ChromaDB / RAG)."""
    from langchain_ollama import OllamaEmbeddings

    resolved_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    resolved_model = model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)

    return OllamaEmbeddings(
        base_url=resolved_url,
        model=resolved_model,
    )


# ──────────────────────────── Health check ─────────────────────────────────


def check_ollama_health(base_url: str | None = None) -> bool:
    """Check whether the Ollama server is reachable."""
    import urllib.error
    import urllib.request

    url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except (urllib.error.URLError, OSError) as e:
        logger.warning(f"Ollama unreachable at {url}: {e}")
        return False


def list_local_models(base_url: str | None = None) -> list[str]:
    """Retrieve the list of models already pulled in Ollama."""
    import json
    import urllib.request

    url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.warning(f"Unable to list Ollama models: {e}")
        return []


# ──────────────────────────── LangFuse ─────────────────────────────────────


def get_langfuse_handler():
    """Return the LangFuse callback handler, or None if unconfigured."""
    try:
        from langfuse.callback import CallbackHandler

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")

        if not public_key or not secret_key:
            logger.info("LangFuse not configured (missing keys), monitoring disabled")
            return None

        return CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as e:
        logger.warning(f"LangFuse unavailable ({e}), monitoring disabled")
        return None
