"""
utils/llm.py — LLM Configuration (Ollama only) + LangFuse monitoring.

Supported models via Ollama:
  - mistral (default)
  - llama3, llama3.1, llama3.2
  - deepseek-coder
  - phi3
  - etc.

Environment variables (.env):
  OLLAMA_BASE_URL    = http://localhost:11434  (default)
  OLLAMA_MODEL       = mistral                 (default)
  OLLAMA_EMBED_MODEL = nomic-embed-text        (default)
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY
  LANGFUSE_HOST      = https://cloud.langfuse.com (default)
"""

import logging
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────── Default config ───────────────────────────────

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

AVAILABLE_MODELS = [
    "mistral",
    "llama3",
    "llama3.1",
    "llama3.2",
    "deepseek-coder",
    "phi3",
    "gemma2",
    "qwen2.5",
    "codellama",
]


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


# ──────────────────────────── LLM ──────────────────────────────────────────


def get_llm(
    temperature: float = 0.3,
    model: str | None = None,
    base_url: str | None = None,
) -> ChatOllama:
    """
    Return a ChatOllama instance.

    Args:
        temperature: Model creativity (0.0 = deterministic, 1.0 = creative).
        model: Ollama model name (overrides OLLAMA_MODEL from .env).
        base_url: Ollama server URL (overrides OLLAMA_BASE_URL from .env).

    Returns:
        Ready-to-use ChatOllama instance.
    """
    resolved_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    resolved_model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    logger.info(f"LLM: {resolved_model} @ {resolved_url} (temp={temperature})")

    return ChatOllama(
        base_url=resolved_url,
        model=resolved_model,
        temperature=temperature,
    )


# ──────────────────────────── Embeddings ───────────────────────────────────


def get_embeddings(
    model: str | None = None,
    base_url: str | None = None,
) -> OllamaEmbeddings:
    """Return the Ollama embeddings model (for ChromaDB / RAG)."""
    resolved_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    resolved_model = model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)

    return OllamaEmbeddings(
        base_url=resolved_url,
        model=resolved_model,
    )


# ──────────────────────────── LangFuse ─────────────────────────────────────


def get_langfuse_handler():
    """
    Create the LangFuse callback handler for monitoring.
    Returns None if LangFuse is not configured or unavailable.
    """
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


# ──────────────────────────── Agent helper ─────────────────────────────────


def invoke_llm(messages: list, temperature: float = 0.3, model: str | None = None):
    """
    Shortcut: invoke the LLM with integrated LangFuse callback.

    Args:
        messages: List of LangChain messages (SystemMessage, HumanMessage, ...).
        temperature: Model temperature.
        model: Ollama model to use (optional).

    Returns:
        The text content of the response (str).
    """
    llm = get_llm(temperature=temperature, model=model)
    handler = get_langfuse_handler()

    config = {}
    if handler:
        config["callbacks"] = [handler]

    response = llm.invoke(messages, config=config)
    return response.content
