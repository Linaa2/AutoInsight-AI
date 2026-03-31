"""utils/langfuse_client.py — LangFuse observability layer for AutoInsight AI.

This is the **single authoritative module** for all LangFuse interactions.
No other module should import ``langfuse`` directly — use the ``monitor``
singleton exported from this file instead.

Architecture
------------
* :class:`LangFuseMonitor` — the main class.  Lazily initialises the
  LangFuse client on first use and provides clean helper methods.
* :data:`monitor` — module-level singleton; import this everywhere.
* :class:`_NoOpSpan` — silent null-object returned when LangFuse is
  disabled or the server is unreachable, so callers never need to
  branch on availability.

Usage::

    from utils.langfuse_client import monitor

    # At the start of an analysis run:
    trace_id = monitor.begin_run(
        dataset_id="sales.csv",
        file_name="sales.csv",
        run_id="abc-123",
    )

    # Inside a graph node (get LangChain callbacks for LLM tracing):
    callbacks = monitor.get_llm_callbacks()

    # Wrap a node execution in a LangFuse span:
    with monitor.node_span("profiler", metadata={"rows": 1000}):
        result = agent.describe(profile, callbacks=callbacks)

    # Log a discrete memory / RAG event:
    monitor.log_event("rag-store-profile", output={"chunks": 5})

    # After the run:
    monitor.end_run(status="success")
    monitor.flush()

Graceful degradation
--------------------
Every public method is wrapped in a broad ``except`` so LangFuse errors
never surface to the application.  When ``LANGFUSE_ENABLED=false`` or keys
are missing, all methods return safe empty values and the ``node_span``
context manager yields a :class:`_NoOpSpan`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from dotenv import dotenv_values

if TYPE_CHECKING:
    from collections.abc import Generator

from config.settings import REPO_ROOT, settings

logger = logging.getLogger(__name__)
_LOCAL_LANGFUSE_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class _LangFuseConfig:
    """Resolved LangFuse credentials for one candidate configuration."""

    source: str
    base_url: str
    public_key: str
    secret_key: str


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def is_langfuse_enabled() -> bool:
    """Return ``True`` when LangFuse is fully configured and enabled.

    The master switch must be enabled and at least one credential source
    must be available:
    * ``LANGFUSE_ENABLED=true`` in env / ``.env``
    * either explicit ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are
      set, or repo-managed local bootstrap credentials exist in
      ``.env.langfuse``
    """
    return bool(_iter_langfuse_configs())


def _key_prefix(value: str) -> str:
    """Return a short redacted key prefix for logging."""
    return f"{value[:12]}..." if value else "<missing>"


def _is_local_langfuse_base_url(base_url: str) -> bool:
    """Return True when *base_url* targets a loopback LangFuse instance."""
    try:
        hostname = urlparse(base_url).hostname or ""
    except Exception:
        return False
    return hostname in _LOCAL_LANGFUSE_HOSTS


def _load_local_bootstrap_config(base_url: str) -> _LangFuseConfig | None:
    """Load repo-managed local LangFuse credentials from ``.env.langfuse``."""
    if not _is_local_langfuse_base_url(base_url):
        return None

    env_path = Path(REPO_ROOT) / ".env.langfuse"
    if not env_path.exists():
        return None

    values = dotenv_values(env_path)
    public_key = str(values.get("LANGFUSE_INIT_PROJECT_PUBLIC_KEY") or "").strip()
    secret_key = str(values.get("LANGFUSE_INIT_PROJECT_SECRET_KEY") or "").strip()
    if not public_key or not secret_key:
        return None

    return _LangFuseConfig(
        source=".env.langfuse",
        base_url=base_url,
        public_key=public_key,
        secret_key=secret_key,
    )


def _iter_langfuse_configs() -> list[_LangFuseConfig]:
    """Return LangFuse credential candidates in the order they should be tried."""
    if not settings.LANGFUSE_ENABLED:
        return []

    base_url = settings.LANGFUSE_BASE_URL.strip()
    if not base_url:
        return []

    candidates: list[_LangFuseConfig] = []
    local_bootstrap = _load_local_bootstrap_config(base_url)
    if local_bootstrap is not None:
        candidates.append(local_bootstrap)

    public_key = settings.LANGFUSE_PUBLIC_KEY.strip()
    secret_key = settings.LANGFUSE_SECRET_KEY.strip()
    if public_key and secret_key:
        env_config = _LangFuseConfig(
            source="environment",
            base_url=base_url,
            public_key=public_key,
            secret_key=secret_key,
        )
        if env_config not in candidates:
            candidates.append(env_config)

    return candidates


def _has_conflicting_langfuse_configs(candidates: list[_LangFuseConfig]) -> bool:
    """Return True when candidate sources disagree on the effective credentials."""
    unique_credentials = {
        (config.base_url, config.public_key, config.secret_key) for config in candidates
    }
    return len(unique_credentials) > 1


def _is_invalid_credentials_error(exc: Exception) -> bool:
    """Return True when *exc* looks like an auth failure."""
    msg = str(exc).lower()
    return any(token in msg for token in ("401", "unauthorized", "invalid credentials"))


# ---------------------------------------------------------------------------
# Null-object span (returned when LangFuse is disabled / unavailable)
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Silent null-object that satisfies the span protocol without side effects."""

    @property
    def id(self) -> str:
        return ""

    def update(self, **kwargs: Any) -> None:
        pass

    def end(self, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------


class LangFuseMonitor:
    """Singleton wrapper around the LangFuse client.

    All LangFuse interactions — trace creation, span management, event
    logging, callback handler creation — go through this class.

    When LangFuse is disabled or unavailable the methods return empty /
    no-op values so the application continues without modification.
    """

    def __init__(self) -> None:
        self._client: Any = None  # Langfuse direct client
        self._client_config: _LangFuseConfig | None = None
        self._trace_id: str = ""  # active run trace ID (empty = no active run)
        self._initialized: bool = False  # guard for lazy init
        self._client_error_kind: str = ""  # "", "auth", or "unavailable"

    # ── Initialisation ─────────────────────────────────────────────────────

    def _auth_check(self, client: Any, config: _LangFuseConfig) -> bool:
        """Return True when *client* is authenticated for *config*."""
        auth_check = getattr(client, "auth_check", None)
        if auth_check is None:
            return True

        try:
            return bool(auth_check())
        except Exception as exc:
            if _is_invalid_credentials_error(exc):
                logger.warning(
                    "LangFuse auth rejected %s credentials (%s) for %s.",
                    config.source,
                    _key_prefix(config.public_key),
                    config.base_url,
                )
                return False
            raise

    def _ensure_client(self) -> Any | None:
        """Lazily create the Langfuse client on first call."""
        if self._initialized:
            return self._client
        self._initialized = True
        self._client_error_kind = ""

        if not is_langfuse_enabled():
            return None

        try:
            from langfuse import Langfuse

            # LangFuse's propagation module (langfuse/_client/propagation.py) logs
            # WARNING when LangGraph injects non-string run metadata into LangChain
            # callbacks (langgraph_step: int, langgraph_triggers: list, etc.).
            # LangFuse silently drops those values — the warnings are harmless noise.
            # We must suppress AFTER the import because langfuse/logger.py resets
            # the level to WARNING when the package is first imported.
            logging.getLogger("langfuse").setLevel(logging.ERROR)
            logging.getLogger("opentelemetry.sdk.trace").setLevel(logging.ERROR)
        except Exception as exc:
            logger.warning("LangFuse initialization failed (running without it): %s", exc)
            self._client = None
            self._client_config = None
            self._client_error_kind = "unavailable"
            return None

        candidates = _iter_langfuse_configs()
        if _has_conflicting_langfuse_configs(candidates):
            logger.info(
                "Detected conflicting LangFuse credential sources for %s; using repo-managed "
                "credentials from .env.langfuse for the local self-hosted stack.",
                settings.LANGFUSE_BASE_URL,
            )

        for config in candidates:
            try:
                # base_url= occupies the highest-priority slot in the SDK's resolution
                # chain (client.py: base_url > LANGFUSE_BASE_URL env > host > LANGFUSE_HOST env).
                # Passing it explicitly ensures our configured value is always used,
                # regardless of any ambient LANGFUSE_BASE_URL in the OS environment.
                candidate = Langfuse(
                    public_key=config.public_key,
                    secret_key=config.secret_key,
                    base_url=config.base_url,
                )
                if not self._auth_check(candidate, config):
                    self._client_error_kind = "auth"
                    continue

                self._client = candidate
                self._client_config = config
                logger.info(
                    "LangFuse initialized — base_url: %s (source: %s, key: %s)",
                    config.base_url,
                    config.source,
                    _key_prefix(config.public_key),
                )
                self._client_error_kind = ""
                return self._client
            except Exception as exc:
                if _is_invalid_credentials_error(exc):
                    self._client_error_kind = "auth"
                    logger.warning(
                        "LangFuse initialization rejected %s credentials (%s) for %s.",
                        config.source,
                        _key_prefix(config.public_key),
                        config.base_url,
                    )
                    continue
                logger.warning(
                    "LangFuse initialization failed for %s (running without it): %s",
                    config.source,
                    exc,
                )
                self._client_error_kind = "unavailable"
                continue

        self._client = None
        self._client_config = None
        return self._client

    # ── Run-level trace ────────────────────────────────────────────────────

    def begin_run(
        self,
        *,
        dataset_id: str = "",  # noqa: ARG002 — kept for API compatibility
        file_name: str = "",  # noqa: ARG002 — kept for API compatibility
        run_id: str = "",
    ) -> str:
        """Start a top-level LangFuse trace for one full analysis run.

        Should be called once per ``run_analysis`` / ``stream_analysis``
        invocation (from ``_make_initial_state``).

        Args:
            dataset_id: The stable dataset identifier.
            file_name:  The original uploaded file name.
            run_id:     A short unique ID for this run (correlation key).

        Returns:
            The LangFuse trace ID, or ``""`` if LangFuse is disabled.
        """
        if not is_langfuse_enabled():
            return ""

        # Initialize the client best-effort (also applies log suppression on
        # first import).  Client being None is acceptable — trace ID generation
        # is a local operation that does NOT require server connectivity.
        self._ensure_client()
        if self._client_error_kind == "auth":
            logger.warning(
                "LangFuse is enabled but the configured credentials were rejected by %s. "
                "Tracing is disabled until the credential source is corrected.",
                settings.LANGFUSE_BASE_URL,
            )
            self._trace_id = ""
            return ""

        try:
            # create_trace_id is a @staticmethod — no client instance needed.
            # It generates a 32-char hex UUID locally; the server is never
            # contacted here, so 401 / unreachable-server errors cannot block it.
            from langfuse import Langfuse as _Langfuse

            self._trace_id = _Langfuse.create_trace_id(seed=run_id or None)
            logger.debug("LangFuse trace ID: %s", self._trace_id)
            return self._trace_id
        except Exception as exc:
            logger.warning("LangFuse trace ID generation failed: %s", exc)
            self._trace_id = ""
            return ""

    def end_run(self, *, status: str = "success", output: dict[str, Any] | None = None) -> None:  # noqa: ARG002
        """Finalize the current run trace with status and summary output.

        In LangFuse v4 the trace lifecycle is managed via OpenTelemetry context;
        this method clears the internal trace ID to signal end of run.

        Args:
            status: ``"success"`` | ``"error"`` | ``"cancelled"``
            output: Dict of summary key-values (e.g. nodes run, errors).
        """
        self._trace_id = ""

    # ── LLM callbacks (LangChain integration) ─────────────────────────────

    def get_llm_callbacks(self) -> list[Any]:
        """Return a LangChain callback list for LLM-level tracing.

        The returned list contains a single ``CallbackHandler`` linked to the
        current run trace, so every LLM call is automatically captured as a
        *generation* child of the trace.

        Pass the list directly to ``chain.invoke(config={"callbacks": ...})``
        or ``llm.invoke(..., config={"callbacks": ...})``.

        Returns:
            ``[CallbackHandler]`` or ``[]`` when LangFuse is disabled.
        """
        client = self._ensure_client()
        config = self._client_config
        if client is None or config is None or not self._trace_id:
            return []
        try:
            # v4 API: link to existing trace via trace_context
            from langfuse.langchain import CallbackHandler

            return [
                CallbackHandler(
                    public_key=config.public_key,
                    trace_context={"trace_id": self._trace_id},
                )
            ]
        except Exception as exc:
            logger.debug("LangFuse get_llm_callbacks failed: %s", exc)
            return []

    # ── Node spans ─────────────────────────────────────────────────────────

    @contextmanager
    def node_span(
        self,
        node_name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[_NoOpSpan, None, None]:
        """Context manager that wraps a graph node in a LangFuse span.

        Yields a :class:`_NoOpSpan` when LangFuse is disabled so callers
        never need to check availability.

        Args:
            node_name: Name of the graph node (e.g. ``"profiler"``).
            metadata:  Optional dict attached to the span (dataset_id, etc.).

        Usage::

            with monitor.node_span("profiler", metadata={"rows": 1000}):
                markdown = agent.describe(profile, callbacks=callbacks)
        """
        client = self._ensure_client()
        if client is None or not self._trace_id:
            yield _NoOpSpan()
            return

        try:
            # v4: start_as_current_observation sets the OTel context so that
            # LangChain callbacks (get_llm_callbacks) are automatically nested
            # under this span.
            ctx = client.start_as_current_observation(
                name=node_name,
                as_type="span",
                metadata=metadata or {},
                trace_context={"trace_id": self._trace_id},
            )
        except Exception as exc:
            logger.debug("LangFuse node_span start failed: %s", exc)
            yield _NoOpSpan()
            return

        with ctx as span:
            yield span

    # ── Discrete events (memory / RAG) ─────────────────────────────────────

    def log_event(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",  # noqa: ARG002 — accepted for API compatibility, v4 span uses level differently
    ) -> None:
        """Log a discrete event to the current trace.

        Lighter than a span — use for instantaneous events (RAG store/retrieve,
        dataset load, etc.) that have no meaningful duration to track.

        Args:
            name:     Event name (e.g. ``"rag-store-profile"``).
            input:    Input context dict.
            output:   Output context dict.
            metadata: Additional key-value metadata.
            level:    LangFuse level: ``"DEFAULT"`` | ``"DEBUG"`` | ``"WARNING"``
                      | ``"ERROR"``.
        """
        client = self._ensure_client()
        if client is None or not self._trace_id:
            return
        try:
            # v4: use a short-lived span as a discrete event
            obs = client.start_observation(
                name=name,
                as_type="span",
                input=input or {},
                output=output or {},
                metadata=metadata or {},
                trace_context={"trace_id": self._trace_id},
            )
            obs.end()
        except Exception as exc:
            logger.debug("LangFuse log_event '%s' failed: %s", name, exc)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush all pending LangFuse events to the server.

        Should be called once at the end of each analysis run (after
        ``end_run``).  Safe to call when LangFuse is disabled.
        """
        if self._client is None:
            return
        try:
            self._client.flush()
            logger.debug("LangFuse flushed")
        except Exception as exc:
            logger.debug("LangFuse flush failed: %s", exc)

    def reset(self) -> None:
        """Reset client state (useful for testing)."""
        self._client = None
        self._client_config = None
        self._trace_id = ""
        self._initialized = False
        self._client_error_kind = ""


# ---------------------------------------------------------------------------
# Module-level singleton — import ``monitor`` everywhere
# ---------------------------------------------------------------------------

#: The single LangFuse monitor instance for the whole process.
monitor = LangFuseMonitor()
