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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def is_langfuse_enabled() -> bool:
    """Return ``True`` when LangFuse is fully configured and enabled.

    All three conditions must hold:
    * ``LANGFUSE_ENABLED=true`` in env / ``.env``
    * ``LANGFUSE_PUBLIC_KEY`` is non-empty
    * ``LANGFUSE_SECRET_KEY`` is non-empty
    """
    return (
        settings.LANGFUSE_ENABLED
        and bool(settings.LANGFUSE_PUBLIC_KEY)
        and bool(settings.LANGFUSE_SECRET_KEY)
    )


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
        self._trace_id: str = ""  # active run trace ID (empty = no active run)
        self._initialized: bool = False  # guard for lazy init

    # ── Initialisation ─────────────────────────────────────────────────────

    def _ensure_client(self) -> Any | None:
        """Lazily create the Langfuse client on first call."""
        if self._initialized:
            return self._client
        self._initialized = True

        if not is_langfuse_enabled():
            return None

        try:
            from langfuse import Langfuse

            # host= is deprecated in v4 but still accepted; base_url= is preferred
            self._client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            logger.info("LangFuse initialized — host: %s", settings.LANGFUSE_HOST)
        except Exception as exc:
            logger.warning("LangFuse initialization failed (running without it): %s", exc)
            self._client = None

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
        client = self._ensure_client()
        if client is None:
            return ""

        try:
            # v4 API: generate a deterministic trace ID from the run_id seed
            self._trace_id = client.create_trace_id(seed=run_id or None)
            logger.debug("LangFuse trace started: %s", self._trace_id)
            return self._trace_id
        except Exception as exc:
            logger.debug("LangFuse begin_run failed: %s", exc)
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
        if not self._trace_id:
            return []
        try:
            # v4 API: link to existing trace via trace_context
            from langfuse.langchain import CallbackHandler

            return [CallbackHandler(trace_context={"trace_id": self._trace_id})]
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
        self._trace_id = ""
        self._initialized = False


# ---------------------------------------------------------------------------
# Module-level singleton — import ``monitor`` everywhere
# ---------------------------------------------------------------------------

#: The single LangFuse monitor instance for the whole process.
monitor = LangFuseMonitor()
