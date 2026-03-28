import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_gemini_healthy: bool | None = None


def _check_gemini() -> bool:
    global _gemini_healthy
    if _gemini_healthy is not None:
        return _gemini_healthy
    try:
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=int(os.getenv("LLM_TIMEOUT", 15)),
        )
        llm.invoke("ping")
        _gemini_healthy = True
        logger.info("LLM primaire : Gemini opérationnel")
    except Exception as e:
        _gemini_healthy = False
        logger.warning(f"Gemini indisponible ({e}), bascule sur Ollama")
    return _gemini_healthy


def get_llm(temperature: float = 0.3):
    if _check_gemini():
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "mistral"),
        temperature=temperature,
    )


def get_langfuse_handler():
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as e:
        logger.warning(f"LangFuse indisponible ({e}), monitoring désactivé")
        return None