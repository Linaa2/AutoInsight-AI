"""Telemetry collection for AutoInsight-AI pipeline nodes.

Each node calls :func:`collect_resource_snapshot` before and after its
core work to capture CPU, memory, and (optionally) GPU utilisation.
:func:`build_telemetry` then packages the two snapshots, model metadata,
and token counts into a :class:`NodeTelemetry` dict that gets stored
inside the :class:`~orchestration.state.NodeTraceEntry`.

The module is intentionally dependency-light — it only uses ``psutil``
(already a transitive dependency) and stdlib.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# Typed structures
# ---------------------------------------------------------------------------


class ResourceSnapshot(TypedDict, total=False):
    """Point-in-time resource measurement."""

    cpu_percent: float  # 0-100 per core, process-level
    memory_rss_mb: float  # resident set size in MiB
    memory_percent: float  # % of total system memory
    # GPU fields — populated only when a GPU backend is detected
    gpu_backend: str | None  # "nvidia" | "mps" | None
    gpu_name: str | None  # human-readable GPU device name
    gpu_memory_mb: float | None  # VRAM in use (MiB)
    gpu_driver_memory_mb: float | None  # driver-allocated VRAM (MPS only)
    gpu_utilization_pct: float | None  # GPU core utilisation % (NVIDIA only)
    disk_read_mb: float  # cumulative disk reads (process)
    disk_write_mb: float  # cumulative disk writes (process)


class ModelInfo(TypedDict, total=False):
    """Metadata about the LLM used by a node."""

    model_name: str
    provider: str  # "ollama" | "gemini"
    is_local: bool
    base_url: str
    timeout_s: int
    task: str  # "text" | "code"


class TokenUsage(TypedDict, total=False):
    """Token-level accounting for an LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class NodeTelemetry(TypedDict, total=False):
    """Rich telemetry payload attached to each NodeTraceEntry."""

    model_info: ModelInfo
    token_usage: TokenUsage
    latency_s: float  # wall-clock time of the LLM call only
    throughput_tok_s: float  # completion_tokens / latency_s
    resource_before: ResourceSnapshot
    resource_after: ResourceSnapshot
    resource_delta: dict[str, float]  # after - before for key metrics


# ---------------------------------------------------------------------------
# Resource collection
# ---------------------------------------------------------------------------

_MiB = 1024 * 1024


def collect_resource_snapshot() -> ResourceSnapshot:
    """Capture a point-in-time resource snapshot for the current process."""
    snap: ResourceSnapshot = {}
    try:
        import psutil

        proc = psutil.Process(os.getpid())

        # CPU — measure over a tiny interval so we get a nonzero reading
        snap["cpu_percent"] = proc.cpu_percent(interval=0.05)

        mem = proc.memory_info()
        snap["memory_rss_mb"] = round(mem.rss / _MiB, 1)
        snap["memory_percent"] = round(proc.memory_percent(), 2)

        try:
            io = proc.io_counters()
            snap["disk_read_mb"] = round(io.read_bytes / _MiB, 2)
            snap["disk_write_mb"] = round(io.write_bytes / _MiB, 2)
        except (AttributeError, psutil.Error):
            pass  # io_counters unavailable on some platforms

    except ImportError:
        pass  # psutil not installed — return empty snapshot

    # GPU (best-effort: NVIDIA via pynvml → Apple MPS via torch)
    _gpu_found = False

    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        snap["gpu_backend"] = "nvidia"
        snap["gpu_name"] = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        snap["gpu_memory_mb"] = round(mem_info.used / _MiB, 1)
        snap["gpu_driver_memory_mb"] = None
        snap["gpu_utilization_pct"] = float(util.gpu)
        pynvml.nvmlShutdown()
        _gpu_found = True
    except Exception:
        pass

    if not _gpu_found:
        # Apple Silicon — Metal Performance Shaders
        try:
            import platform

            import torch  # type: ignore[import-untyped]

            if torch.backends.mps.is_available():
                chip = platform.processor() or "arm64"
                snap["gpu_backend"] = "mps"
                snap["gpu_name"] = f"Apple GPU (MPS · {chip})"
                snap["gpu_memory_mb"] = round(torch.mps.current_allocated_memory() / _MiB, 1)
                snap["gpu_driver_memory_mb"] = round(torch.mps.driver_allocated_memory() / _MiB, 1)
                snap["gpu_utilization_pct"] = None  # not exposed by MPS API
                _gpu_found = True
        except Exception:
            pass

    if not _gpu_found:
        snap["gpu_backend"] = None
        snap["gpu_name"] = None
        snap["gpu_memory_mb"] = None
        snap["gpu_driver_memory_mb"] = None
        snap["gpu_utilization_pct"] = None

    return snap


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


def get_model_info(task: str = "text") -> ModelInfo:
    """Build :class:`ModelInfo` from current settings.

    Args:
        task: ``"text"`` or ``"code"`` — determines which model name is used.
    """
    from config.settings import settings

    is_local = settings.LLM_PROVIDER != "gemini"
    model_name = settings.OLLAMA_TEXT_MODEL if task == "text" else settings.OLLAMA_CODE_MODEL
    if not is_local:
        model_name = settings.GEMINI_MODEL

    return ModelInfo(
        model_name=model_name,
        provider=settings.LLM_PROVIDER,
        is_local=is_local,
        base_url=settings.OLLAMA_BASE_URL
        if is_local
        else "https://generativelanguage.googleapis.com",
        timeout_s=settings.LLM_TIMEOUT,
        task=task,
    )


# ---------------------------------------------------------------------------
# Token extraction helpers
# ---------------------------------------------------------------------------


def extract_token_usage(llm_response: Any) -> TokenUsage:
    """Best-effort extraction of token counts from a LangChain response.

    Works with:
    - ``AIMessage.usage_metadata`` (LangChain ≥ 0.2)
    - ``AIMessage.response_metadata["token_usage"]``
    - Ollama-style ``response_metadata``
    """
    usage: TokenUsage = {}

    if llm_response is None:
        return usage

    # LangChain ≥ 0.2 unified usage_metadata
    meta = getattr(llm_response, "usage_metadata", None)
    if meta and isinstance(meta, dict):
        usage["prompt_tokens"] = meta.get("input_tokens", 0)
        usage["completion_tokens"] = meta.get("output_tokens", 0)
        usage["total_tokens"] = meta.get("total_tokens", 0)
        return usage

    # Fallback: response_metadata
    resp_meta = getattr(llm_response, "response_metadata", None) or {}

    # Ollama format
    if "prompt_eval_count" in resp_meta:
        usage["prompt_tokens"] = resp_meta.get("prompt_eval_count", 0)
        usage["completion_tokens"] = resp_meta.get("eval_count", 0)
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        return usage

    # OpenAI / Gemini format
    tok = resp_meta.get("token_usage") or resp_meta.get("usage", {})
    if tok:
        usage["prompt_tokens"] = tok.get("prompt_tokens", 0)
        usage["completion_tokens"] = tok.get("completion_tokens", 0)
        usage["total_tokens"] = tok.get("total_tokens", 0)

    return usage


# ---------------------------------------------------------------------------
# Telemetry assembly
# ---------------------------------------------------------------------------


def build_telemetry(
    *,
    task: str,
    llm_start: float,
    llm_end: float,
    llm_response: Any = None,
    resource_before: ResourceSnapshot,
    resource_after: ResourceSnapshot,
) -> NodeTelemetry:
    """Assemble the full telemetry dict for a node execution.

    Args:
        task: ``"text"`` or ``"code"``.
        llm_start: ``time.time()`` captured right before the LLM call.
        llm_end: ``time.time()`` captured right after the LLM call.
        llm_response: The raw LangChain ``AIMessage`` (``chain.invoke()`` result).
        resource_before: Snapshot taken before node work.
        resource_after: Snapshot taken after node work.
    """
    latency = round(llm_end - llm_start, 3)
    tokens = extract_token_usage(llm_response)
    completion = tokens.get("completion_tokens", 0)

    delta: dict[str, float] = {}
    for key in ("cpu_percent", "memory_rss_mb", "memory_percent"):
        before_val = float(resource_before.get(key) or 0)  # type: ignore[arg-type]
        after_val = float(resource_after.get(key) or 0)  # type: ignore[arg-type]
        delta[key] = round(after_val - before_val, 2)

    telem = NodeTelemetry(
        model_info=get_model_info(task),
        token_usage=tokens,
        latency_s=latency,
        throughput_tok_s=round(completion / latency, 1) if latency > 0 and completion else 0,
        resource_before=resource_before,
        resource_after=resource_after,
        resource_delta=delta,
    )
    return telem


def build_telemetry_no_llm(
    *,
    resource_before: ResourceSnapshot,
    resource_after: ResourceSnapshot,
) -> NodeTelemetry:
    """Build a minimal telemetry dict for nodes that include deterministic work."""
    delta: dict[str, float] = {}
    for key in ("cpu_percent", "memory_rss_mb", "memory_percent"):
        before_val = float(resource_before.get(key) or 0)  # type: ignore[arg-type]
        after_val = float(resource_after.get(key) or 0)  # type: ignore[arg-type]
        delta[key] = round(after_val - before_val, 2)

    return NodeTelemetry(
        resource_before=resource_before,
        resource_after=resource_after,
        resource_delta=delta,
    )
