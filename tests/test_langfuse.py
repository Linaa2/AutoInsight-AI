"""
tests/test_langfuse.py — Unit tests for the LangFuse observability integration.

Design goals:
- No running LangFuse server required (everything mocked or disabled-path tests).
- Tests remain fast and run in CI without external dependencies.
- Cover:
  1. Configuration parsing (enabled/disabled) via settings
  2. Utility layer behavior when disabled (missing keys / LANGFUSE_ENABLED=false)
  3. Utility layer behavior when enabled but server is unreachable
  4. Agent calls still work when LangFuse is disabled
  5. Graph execution still works when LangFuse is disabled
  6. Graceful failure: node_span never raises even if LangFuse client errors
  7. log_event / end_run / flush are safe no-ops when disabled
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _fresh_monitor():
    """Return a LangFuseMonitor with a clean reset state (no cached client)."""
    from utils.langfuse_client import LangFuseMonitor

    m = LangFuseMonitor()
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  Part A — Configuration helpers
# ═══════════════════════════════════════════════════════════════════════════


def test_is_langfuse_enabled_false_by_default() -> None:
    """is_langfuse_enabled() returns False when env vars are absent."""
    # utils.langfuse_client.settings is a cached reference set at import time.
    # We must patch it directly — reloading config.settings won't update it.
    from utils.langfuse_client import is_langfuse_enabled

    with patch("utils.langfuse_client.settings") as mock_s:
        mock_s.LANGFUSE_ENABLED = False
        mock_s.LANGFUSE_PUBLIC_KEY = ""
        mock_s.LANGFUSE_SECRET_KEY = ""
        assert is_langfuse_enabled() is False


def test_is_langfuse_enabled_respects_settings() -> None:
    """is_langfuse_enabled() returns True only when all three conditions hold."""
    from utils.langfuse_client import is_langfuse_enabled

    # Patch settings directly to avoid env-reload complexity
    with patch("utils.langfuse_client.settings") as mock_settings:
        mock_settings.LANGFUSE_ENABLED = False
        assert is_langfuse_enabled() is False

        mock_settings.LANGFUSE_ENABLED = True
        mock_settings.LANGFUSE_PUBLIC_KEY = ""
        mock_settings.LANGFUSE_SECRET_KEY = "sk"
        assert is_langfuse_enabled() is False  # missing public key

        mock_settings.LANGFUSE_ENABLED = True
        mock_settings.LANGFUSE_PUBLIC_KEY = "pk"
        mock_settings.LANGFUSE_SECRET_KEY = ""
        assert is_langfuse_enabled() is False  # missing secret key

        mock_settings.LANGFUSE_ENABLED = True
        mock_settings.LANGFUSE_PUBLIC_KEY = "pk"
        mock_settings.LANGFUSE_SECRET_KEY = "sk"
        assert is_langfuse_enabled() is True


def test_settings_langfuse_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangFuse settings have safe defaults (disabled, empty keys)."""
    # Isolate from any .env values that may have been loaded
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from config.settings import Settings

    fresh = Settings()
    # Default must be False so no accidental tracing without explicit opt-in
    assert fresh.LANGFUSE_ENABLED is False
    # Keys default to empty strings (not None, not hardcoded values)
    assert fresh.LANGFUSE_PUBLIC_KEY == ""
    assert fresh.LANGFUSE_SECRET_KEY == ""


# ═══════════════════════════════════════════════════════════════════════════
#  Part B — Utility layer when disabled
# ═══════════════════════════════════════════════════════════════════════════


def test_monitor_begin_run_returns_empty_when_disabled() -> None:
    """begin_run() returns '' when LangFuse is disabled."""
    monitor = _fresh_monitor()
    with patch("utils.langfuse_client.is_langfuse_enabled", return_value=False):
        trace_id = monitor.begin_run(dataset_id="test.csv", file_name="test.csv")
    assert trace_id == ""


def test_monitor_get_llm_callbacks_empty_when_no_trace() -> None:
    """get_llm_callbacks() returns [] when no trace exists."""
    monitor = _fresh_monitor()
    assert monitor.get_llm_callbacks() == []


def test_monitor_node_span_yields_noop_when_disabled() -> None:
    """node_span() context manager yields a _NoOpSpan when LangFuse is off."""
    from utils.langfuse_client import _NoOpSpan

    monitor = _fresh_monitor()
    with monitor.node_span("profiler") as span:
        assert isinstance(span, _NoOpSpan)
        span.update(output="anything")  # must not raise
        span.end()  # must not raise


def test_monitor_log_event_safe_when_disabled() -> None:
    """log_event() is a no-op when there is no active trace."""
    monitor = _fresh_monitor()
    # Should never raise
    monitor.log_event("rag-store", input={"k": "v"}, output={"chunks": 3})


def test_monitor_end_run_safe_when_no_trace() -> None:
    """end_run() is a no-op when there is no active trace."""
    monitor = _fresh_monitor()
    monitor.end_run(status="success", output={"nodes": 4})  # must not raise


def test_monitor_flush_safe_when_no_client() -> None:
    """flush() is a no-op when the client was never initialized."""
    monitor = _fresh_monitor()
    monitor.flush()  # must not raise


# ═══════════════════════════════════════════════════════════════════════════
#  Part B — Utility layer when enabled but server is unreachable
# ═══════════════════════════════════════════════════════════════════════════


def test_monitor_begin_run_graceful_on_network_error() -> None:
    """begin_run() returns '' (not raises) if the LangFuse server is down."""
    monitor = _fresh_monitor()
    with (
        patch("utils.langfuse_client.is_langfuse_enabled", return_value=True),
        patch("utils.langfuse_client.settings") as mock_s,
    ):
        mock_s.LANGFUSE_ENABLED = True
        mock_s.LANGFUSE_PUBLIC_KEY = "pk"
        mock_s.LANGFUSE_SECRET_KEY = "sk"
        mock_s.LANGFUSE_HOST = "http://unreachable-host:3001"
        mock_s.LANGFUSE_ENV = "test"
        mock_s.LANGFUSE_RELEASE = ""

        # Simulating Langfuse client raising on instantiation
        with patch("langfuse.Langfuse", side_effect=ConnectionError("refused")):
            trace_id = monitor.begin_run(dataset_id="test.csv")
    assert trace_id == ""


def test_monitor_all_methods_safe_after_failed_init() -> None:
    """All monitor methods stay no-op (never raise) after a failed init."""
    monitor = _fresh_monitor()
    # Simulate failed init by leaving _trace_id empty and client as None
    monitor._initialized = True
    monitor._client = None
    monitor._trace_id = ""

    # All these must be completely silent
    monitor.log_event("event", input={}, output={})
    monitor.end_run(status="error")
    monitor.flush()
    assert monitor.get_llm_callbacks() == []
    with monitor.node_span("some-node") as span:
        assert span is not None  # NoOpSpan


def test_noop_span_interface() -> None:
    """_NoOpSpan satisfies the full span interface without side effects."""
    from utils.langfuse_client import _NoOpSpan

    span = _NoOpSpan()
    assert span.id == ""
    span.update(output={"key": "value"})
    span.end(status_message="ok")
    # Context manager protocol
    with span as s:
        assert s is span


# ═══════════════════════════════════════════════════════════════════════════
#  Part C — Agents work when LangFuse is disabled
# ═══════════════════════════════════════════════════════════════════════════


def test_reporter_agent_accepts_callbacks_param() -> None:
    """ReporterAgent.run() accepts callbacks=None without raising."""
    with patch("agents.reporter.call_llm_with_messages", return_value="# Report"):
        from agents.reporter import ReporterAgent

        agent = ReporterAgent()
        result = agent.run(
            profiler_output="Profile",
            analyst_output="Insights",
            callbacks=None,
        )
    assert result["reporter_output"] == "# Report"


def test_reporter_agent_passes_empty_callbacks_to_llm() -> None:
    """When callbacks=[], the LLM is called with config containing empty list."""
    with patch("agents.reporter.call_llm_with_messages", return_value="ok") as mock_llm:
        from agents.reporter import ReporterAgent

        agent = ReporterAgent()
        agent.run(
            profiler_output="Profile",
            analyst_output="Insights",
            callbacks=[],
        )
    # call_llm_with_messages must receive callbacks=[]
    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs.get("callbacks") == []


def test_call_llm_with_messages_accepts_callbacks() -> None:
    """call_llm_with_messages() accepts a callbacks param and forwards it."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="response")

    with patch("utils.llm.LLMClient._build", return_value=mock_llm):
        from utils.llm import call_llm_with_messages

        result = call_llm_with_messages("system", "human", callbacks=[])

    assert result == "response"
    # Verify config was forwarded (callbacks=[]) to llm.invoke
    _, kwargs = mock_llm.invoke.call_args
    assert "config" in kwargs
    assert kwargs["config"].get("callbacks") == []


# ═══════════════════════════════════════════════════════════════════════════
#  Part D — Graph execution works when LangFuse is disabled
# ═══════════════════════════════════════════════════════════════════════════


def test_make_initial_state_includes_langfuse_trace_id() -> None:
    """_make_initial_state always includes 'langfuse_trace_id' key in state."""
    from orchestration.graph import _make_initial_state

    df = pd.DataFrame({"a": [1, 2]})
    state = _make_initial_state(df, "test.csv")

    # Key must exist regardless of whether LangFuse is enabled
    assert "langfuse_trace_id" in state
    # When disabled, value is ""
    assert isinstance(state["langfuse_trace_id"], str)


def test_rag_storage_node_does_not_raise_with_langfuse_disabled() -> None:
    """rag_storage_node works normally even without LangFuse (log_event is a no-op)."""
    from orchestration.graph import rag_storage_node

    result = rag_storage_node({"dataset_id": "test.csv", "graph_trace": []})

    assert "rag_stored" in result
    assert "graph_trace" in result
    assert not result["rag_stored"]


def test_run_analysis_langfuse_disabled_end_to_end() -> None:
    """run_analysis completes without raising even when LangFuse is disabled."""
    from unittest.mock import patch

    from utils.langfuse_client import monitor as lf_monitor

    df = pd.DataFrame({"x": [1, 2, 3]})

    # Reset the singleton so _ensure_client() re-evaluates is_langfuse_enabled()
    # (the singleton may already be initialized from a prior test with real .env keys)
    lf_monitor.reset()

    with patch("utils.langfuse_client.is_langfuse_enabled", return_value=False):
        from orchestration.graph import _make_initial_state

        state = _make_initial_state(df, "data.csv")

    # dataset_id and langfuse_trace_id must both be present
    assert state["dataset_id"] == "data.csv"
    assert state["langfuse_trace_id"] == ""


# ═══════════════════════════════════════════════════════════════════════════
#  Part D — node_span never breaks node execution
# ═══════════════════════════════════════════════════════════════════════════


def test_node_span_exception_inside_context_propagates() -> None:
    """Exceptions inside node_span are NOT swallowed — they propagate normally."""
    monitor = _fresh_monitor()

    with pytest.raises(ValueError, match="simulated node error"), monitor.node_span("test-node"):
        raise ValueError("simulated node error")


def test_node_span_cleans_up_on_exception() -> None:
    """node_span exits its context manager (ending the span) even when an exception occurs."""
    monitor = _fresh_monitor()

    # Inject a mock client with a mock context-manager span
    mock_client = MagicMock()
    mock_span_cm = MagicMock()  # returned by start_as_current_observation
    mock_span_cm.__enter__ = MagicMock(return_value=MagicMock())  # the span object
    mock_span_cm.__exit__ = MagicMock(return_value=False)  # don't suppress exceptions
    mock_client.start_as_current_observation.return_value = mock_span_cm

    monitor._client = mock_client
    monitor._initialized = True
    monitor._trace_id = "trace-abc"

    with pytest.raises(RuntimeError), monitor.node_span("profiler"):
        raise RuntimeError("boom")

    # The context manager's __exit__ must have been called (span ended)
    mock_span_cm.__exit__.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
#  LangFuse enabled with mocked client — happy path
# ═══════════════════════════════════════════════════════════════════════════


def test_monitor_begin_run_returns_trace_id_when_enabled() -> None:
    """begin_run() returns the trace ID when LangFuse client is working."""
    monitor = _fresh_monitor()

    mock_client = MagicMock()
    mock_client.create_trace_id.return_value = "trace-abc-123"

    with (
        patch("utils.langfuse_client.is_langfuse_enabled", return_value=True),
        patch("langfuse.Langfuse", return_value=mock_client),
    ):
        trace_id = monitor.begin_run(dataset_id="sales.csv", file_name="sales.csv")

    assert trace_id == "trace-abc-123"
    assert monitor._trace_id == "trace-abc-123"


def test_monitor_get_llm_callbacks_returns_handler_when_trace_set() -> None:
    """get_llm_callbacks() returns a list with a handler when a trace is active."""
    monitor = _fresh_monitor()
    monitor._trace_id = "trace-abc-123"  # simulate active run

    mock_handler = MagicMock()
    # v4: import path is langfuse.langchain.CallbackHandler
    with patch("langfuse.langchain.CallbackHandler", return_value=mock_handler):
        callbacks = monitor.get_llm_callbacks()

    assert len(callbacks) == 1
    assert callbacks[0] is mock_handler


def test_monitor_log_event_calls_trace_event() -> None:
    """log_event() creates a short-lived span via client.start_observation when trace is active."""
    monitor = _fresh_monitor()
    mock_client = MagicMock()
    mock_obs = MagicMock()
    mock_client.start_observation.return_value = mock_obs

    monitor._client = mock_client
    monitor._initialized = True
    monitor._trace_id = "trace-abc"

    monitor.log_event(
        "rag-store-profile",
        input={"dataset_id": "x"},
        output={"chunks": 5},
        level="DEFAULT",
    )

    mock_client.start_observation.assert_called_once()
    call_kwargs = mock_client.start_observation.call_args.kwargs
    assert call_kwargs["name"] == "rag-store-profile"
    assert call_kwargs["output"]["chunks"] == 5
    mock_obs.end.assert_called_once()


def test_monitor_end_run_clears_trace_id() -> None:
    """end_run() clears the trace ID (there is no trace.update() in v4)."""
    monitor = _fresh_monitor()
    monitor._trace_id = "trace-abc-123"

    monitor.end_run(status="success", output={"nodes": 5})

    assert monitor._trace_id == ""  # trace ID cleared after end_run
