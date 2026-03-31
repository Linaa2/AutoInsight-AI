"""Diagnostics tab renderer for AutoInsight-AI.

This module is the **single owner** of the diagnostics UI.  ``app/main.py``
delegates to :func:`render_diagnostics_tab` and never touches trace
rendering itself.

Sections rendered (in order):
1. **Pipeline Architecture Diagram** — React Flow canvas with per-node status.
2. **Aggregated Metrics Summary** — KPIs across all nodes.
3. **Per-Agent Detail Cards** — one expander per unique node, each containing
   timing, model info, token usage, resource consumption, and status.
4. **Memory & RAG** — ChromaDB storage events for this pipeline run.
5. **Download** — full trace as JSON.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import ManualLayout
from streamlit_flow.state import StreamlitFlowState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATUS_BADGE: dict[str, str] = {"success": "✅", "failed": "❌", "skipped": "⏭️"}

_AGENT_META: dict[str, dict[str, str]] = {
    "profiler": {"icon": "📊", "label": "Profiler"},
    "analyst": {"icon": "💡", "label": "Analyst"},
    "uncertainty": {"icon": "🎯", "label": "Confidence"},
    "visualizer": {"icon": "📈", "label": "Visualizer"},
    "reporter": {"icon": "📄", "label": "Reporter"},
    "rag_storage": {"icon": "🧠", "label": "RAG Storage"},
}

_NODE_ORDER = ["profiler", "analyst", "uncertainty", "visualizer", "reporter", "rag_storage"]

# React Flow node colours per status
_STATUS_STYLE: dict[str, dict[str, str]] = {
    "success": {
        "background": "#d1fae5",
        "border": "2px solid #059669",
        "color": "#065f46",
    },
    "failed": {
        "background": "#fee2e2",
        "border": "2px solid #dc2626",
        "color": "#991b1b",
    },
    "skipped": {
        "background": "#fef3c7",
        "border": "2px solid #d97706",
        "color": "#92400e",
    },
    "pending": {
        "background": "#f1f5f9",
        "border": "2px dashed #94a3b8",
        "color": "#475569",
    },
    "running": {
        "background": "#dbeafe",
        "border": "2px solid #3b82f6",
        "color": "#1e40af",
    },
}


# ---------------------------------------------------------------------------
# React Flow pipeline canvas
# ---------------------------------------------------------------------------

# Horizontal positions: START + 5 agents + END, spaced 200 px apart
_NODE_X: dict[str, float] = {
    "start": 0,
    "profiler": 200,
    "analyst": 400,
    "uncertainty": 600,
    "visualizer": 800,
    "reporter": 1000,
    "rag_storage": 1050,
    "end": 1200,
}
_NODE_Y = 80  # vertical centre for all nodes

_SENTINEL_STYLE: dict[str, str] = {
    "background": "#e0e7ff",
    "border": "2px solid #4338ca",
    "color": "#3730a3",
    "borderRadius": "20px",
    "padding": "8px 18px",
    "fontWeight": "700",
    "fontSize": "13px",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.12)",
}


def _agent_flow_node(
    node_id: str,
    label_lines: list[str],
    status: str,
) -> StreamlitFlowNode:
    """Build a single styled React Flow agent node."""
    style = dict(
        _STATUS_STYLE.get(status, _STATUS_STYLE["pending"]),
        borderRadius="10px",
        padding="10px 14px",
        fontFamily="sans-serif",
        fontSize="13px",
        textAlign="center",
        minWidth="130px",
        boxShadow="0 2px 6px rgba(0,0,0,0.12)",
    )
    content = "<br>".join(label_lines)
    return StreamlitFlowNode(
        id=node_id,
        pos=(_NODE_X.get(node_id, 0), _NODE_Y),
        data={"content": content},
        node_type="default",
        source_position="right",
        target_position="left",
        draggable=False,
        selectable=False,
        connectable=False,
        deletable=False,
        style=style,
    )


def _render_pipeline_diagram(
    trace: list[dict[str, Any]],
    current_agent: str = "",
    key_suffix: str = "",
) -> None:
    """Render the pipeline architecture as an interactive React Flow canvas."""
    status_map = {e["node"]: e.get("status", "pending") for e in trace if "node" in e}
    duration_map = {e["node"]: e.get("duration_s", 0.0) for e in trace if "node" in e}

    nodes: list[StreamlitFlowNode] = []
    edges: list[StreamlitFlowEdge] = []

    # START sentinel
    nodes.append(
        StreamlitFlowNode(
            id="start",
            pos=(_NODE_X["start"], _NODE_Y),
            data={"content": "▶ START"},
            node_type="input",
            source_position="right",
            target_position="left",
            draggable=False,
            selectable=False,
            connectable=False,
            deletable=False,
            style=_SENTINEL_STYLE,
        )
    )

    prev_id = "start"
    for agent_id in _NODE_ORDER:
        meta = _AGENT_META[agent_id]
        status = status_map.get(agent_id, "pending")
        # Promote the currently-executing agent from "pending" to "running"
        if agent_id == current_agent and status == "pending":
            status = "running"
        badge = _STATUS_BADGE.get(status, "⏳")
        dur = duration_map.get(agent_id, 0.0)
        dur_str = f"{dur:.1f}s" if status not in {"pending", "running"} else "…"
        label_lines = [
            f"{meta['icon']} <b>{meta['label']}</b>",
            f"{badge} {status.capitalize()} · {dur_str}",
        ]
        nodes.append(_agent_flow_node(agent_id, label_lines, status))
        # Animate edges flowing into nodes that are still active (running / pending)
        edges.append(
            StreamlitFlowEdge(
                id=f"{prev_id}-{agent_id}",
                source=prev_id,
                target=agent_id,
                edge_type="smoothstep",
                animated=(status in {"running", "pending"}),
                marker_end={"type": "arrowclosed"},
                style={"stroke": "#6366f1", "strokeWidth": 2},
            )
        )
        prev_id = agent_id

    # END sentinel
    nodes.append(
        StreamlitFlowNode(
            id="end",
            pos=(_NODE_X["end"], _NODE_Y),
            data={"content": "■ END"},
            node_type="output",
            source_position="right",
            target_position="left",
            draggable=False,
            selectable=False,
            connectable=False,
            deletable=False,
            style=_SENTINEL_STYLE,
        )
    )
    edges.append(
        StreamlitFlowEdge(
            id=f"{prev_id}-end",
            source=prev_id,
            target="end",
            edge_type="smoothstep",
            marker_end={"type": "arrowclosed"},
            style={"stroke": "#6366f1", "strokeWidth": 2},
        )
    )

    flow_state = StreamlitFlowState(nodes=nodes, edges=edges)
    streamlit_flow(
        key=f"pipeline_flow{key_suffix}",
        state=flow_state,
        height=220,
        fit_view=True,
        show_controls=False,
        show_minimap=False,
        allow_new_edges=False,
        pan_on_drag=False,
        allow_zoom=False,
        layout=ManualLayout(),
        hide_watermark=True,
    )


def render_pipeline_diagram(
    trace: list[dict[str, Any]],
    current_agent: str = "",
    key_suffix: str = "",
) -> None:
    """Public API — render the React Flow pipeline architecture canvas.

    Args:
        trace: ``graph_trace`` list from the pipeline state.
        current_agent: Name of the agent currently executing (shown in blue).
        key_suffix: Unique suffix to avoid duplicate Streamlit element IDs.
    """
    _render_pipeline_diagram(trace, current_agent=current_agent, key_suffix=key_suffix)


# ---------------------------------------------------------------------------
# Aggregated Summary
# ---------------------------------------------------------------------------


def _render_aggregated_summary(trace: list[dict[str, Any]]) -> None:
    """Render a top-level metrics summary across all nodes."""
    total_duration = sum(e.get("duration_s", 0) for e in trace)
    n_success = sum(1 for e in trace if e.get("status") == "success")
    n_failed = sum(1 for e in trace if e.get("status") == "failed")
    n_skipped = sum(1 for e in trace if e.get("status") == "skipped")

    # Aggregate token usage across all nodes
    total_prompt = 0
    total_completion = 0
    total_latency = 0.0
    has_tokens = False
    for e in trace:
        telem = e.get("telemetry") or {}
        tok = telem.get("token_usage") or {}
        if tok.get("prompt_tokens") or tok.get("completion_tokens"):
            has_tokens = True
            total_prompt += tok.get("prompt_tokens", 0)
            total_completion += tok.get("completion_tokens", 0)
        total_latency += telem.get("latency_s", 0)

    # Resource peak
    peak_mem = 0.0
    for e in trace:
        telem = e.get("telemetry") or {}
        after = telem.get("resource_after") or {}
        rss = after.get("memory_rss_mb", 0)
        if rss and rss > peak_mem:
            peak_mem = rss

    # Row 1: pipeline-level KPIs
    st.markdown("### 📋 Pipeline Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Duration", f"{total_duration:.1f}s")
    c2.metric("Nodes Run", f"{n_success + n_failed + n_skipped}")
    c3.metric("✅ Succeeded", str(n_success))
    c4.metric("❌ Failed", str(n_failed))
    c5.metric("⏭️ Skipped", str(n_skipped))

    # Row 2: LLM & resource KPIs
    if has_tokens or peak_mem > 0:
        cols = st.columns(5)
        if has_tokens:
            total_tokens = total_prompt + total_completion
            avg_throughput = round(total_completion / total_latency, 1) if total_latency > 0 else 0
            cols[0].metric("Prompt Tokens", f"{total_prompt:,}")
            cols[1].metric("Completion Tokens", f"{total_completion:,}")
            cols[2].metric("Total Tokens", f"{total_tokens:,}")
            cols[3].metric("Avg Throughput", f"{avg_throughput} tok/s")
        else:
            cols[0].metric("Prompt Tokens", "—")
            cols[1].metric("Completion Tokens", "—")
            cols[2].metric("Total Tokens", "—")
            cols[3].metric("Avg Throughput", "—")
        if peak_mem > 0:
            cols[4].metric("Peak Memory", f"{peak_mem:.0f} MiB")
        else:
            cols[4].metric("Peak Memory", "—")


# ---------------------------------------------------------------------------
# Per-node detail cards
# ---------------------------------------------------------------------------


def _render_node_card(entry: dict[str, Any]) -> None:
    """Render one expandable card for a single node."""
    node_name = entry.get("node", "?")
    status = entry.get("status", "unknown")
    meta = _AGENT_META.get(node_name, {"icon": "❓", "label": node_name})
    badge = _STATUS_BADGE.get(status, "❓")
    dur = entry.get("duration_s", 0)

    with st.expander(
        f"{badge} {meta['icon']} **{meta['label']}** — {status}  ·  {dur:.1f}s",
        expanded=(status == "failed"),
    ):
        # ── Timing overview ──
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Status", f"{badge} {status.capitalize()}")
        tc2.metric("Total Duration", f"{dur:.3f}s")
        # Show only HH:MM:SS to keep it compact
        started = entry.get("started_at", "—")
        finished = entry.get("finished_at", "—")
        tc3.metric("Started", started.split("T")[1][:8] if "T" in (started or "") else started)
        tc4.metric("Finished", finished.split("T")[1][:8] if "T" in (finished or "") else finished)

        # ── State I/O (compact) ──
        keys_r = entry.get("keys_read", [])
        keys_w = entry.get("keys_written", [])
        kr = ", ".join(f"`{k}`" for k in keys_r) if keys_r else "—"
        kw = ", ".join(f"`{k}`" for k in keys_w) if keys_w else "—"
        st.caption(f"**Reads:** {kr}")
        st.caption(f"**Writes:** {kw}")

        summary = entry.get("summary")
        if summary:
            st.info(f"📝 {summary}")

        err = entry.get("error")
        if err:
            st.error(err)

        # ── Telemetry ──
        telem = entry.get("telemetry")
        if not telem:
            st.caption("No telemetry collected for this node.")
            return

        st.divider()

        model = telem.get("model_info") or {}
        tok = telem.get("token_usage") or {}
        latency = telem.get("latency_s") or 0.0
        throughput = telem.get("throughput_tok_s") or 0.0

        # ── Model + LLM performance (always shown when telemetry present) ──
        st.markdown("#### 🤖 Model & LLM Performance")
        is_local = model.get("is_local", True)
        provider_icon = "🏠" if is_local else "☁️"
        provider_label = (
            f"{provider_icon} {'Local' if is_local else 'Cloud'} · {model.get('provider', '—')}"
        )

        prompt_tok = tok.get("prompt_tokens", 0) or 0
        comp_tok = tok.get("completion_tokens", 0) or 0
        total_tok = tok.get("total_tokens", 0) or (prompt_tok + comp_tok)
        has_tok = total_tok > 0

        # Row 1: identity
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Model", model.get("model_name", "—"))
        mc2.metric("Provider", provider_label)
        mc3.metric("Task", (model.get("task") or "—").capitalize())
        # Row 2: performance
        mc4, mc5, mc6 = st.columns(3)
        mc4.metric("LLM Latency", f"{latency:.2f}s" if latency else "—")
        mc5.metric("Throughput", f"{throughput:.1f} tok/s" if throughput else "—")
        mc6.metric("Timeout cfg", f"{model.get('timeout_s', '—')}s")

        if not is_local:
            st.caption(f"Endpoint: `{model.get('base_url', '—')}`")

        # ── Token usage (always shown, marks unavailable with '—') ──
        st.markdown("#### 📊 Token Usage")
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Prompt Tokens", f"{prompt_tok:,}" if has_tok else "—")
        lc2.metric("Completion Tokens", f"{comp_tok:,}" if has_tok else "—")
        lc3.metric("Total Tokens", f"{total_tok:,}" if has_tok else "—")
        if not has_tok:
            st.caption(
                "Note: Token counts not reported by this provider. "
                "This is normal for Ollama models — latency is still measured via wall-clock time."
            )

        # ── Resource consumption ──
        after = telem.get("resource_after") or {}
        delta = telem.get("resource_delta") or {}
        if after.get("memory_rss_mb") or after.get("cpu_percent"):
            st.markdown("#### 💻 Resource Consumption")
            rss = after.get("memory_rss_mb") or 0.0
            cpu = after.get("cpu_percent") or 0.0
            mem_pct = after.get("memory_percent") or 0.0
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric(
                "Memory (RSS)",
                f"{rss:.0f} MiB",
                delta=f"{delta.get('memory_rss_mb', 0):+.1f} MiB",
            )
            rc2.metric("CPU Usage", f"{cpu:.1f}%")
            rc3.metric("Memory %", f"{mem_pct:.1f}%")

            gpu_backend = after.get("gpu_backend")
            if gpu_backend in ("nvidia", "mps"):
                st.markdown("#### 🎮 GPU")
                gpu_name = after.get("gpu_name")
                if gpu_name:
                    st.caption(f"**Device:** {gpu_name}")
                gpu_mem = after.get("gpu_memory_mb")
                gpu_util = after.get("gpu_utilization_pct")
                if gpu_backend == "mps":
                    gpu_driver = after.get("gpu_driver_memory_mb")
                    gc1, gc2 = st.columns(2)
                    gc1.metric(
                        "Allocated VRAM",
                        f"{gpu_mem:.0f} MiB" if gpu_mem is not None else "—",
                    )
                    gc2.metric(
                        "Driver VRAM",
                        f"{gpu_driver:.0f} MiB" if gpu_driver is not None else "—",
                    )
                else:  # nvidia
                    gc1, gc2 = st.columns(2)
                    gc1.metric(
                        "GPU Memory",
                        f"{gpu_mem:.0f} MiB" if gpu_mem is not None else "—",
                    )
                    gc2.metric(
                        "GPU Utilization",
                        f"{gpu_util:.0f}%" if gpu_util is not None else "—",
                    )

            dr = after.get("disk_read_mb")
            dw = after.get("disk_write_mb")
            if dr is not None or dw is not None:
                dc1, dc2 = st.columns(2)
                if dr is not None:
                    dc1.metric("Disk Read (cumul.)", f"{dr:.1f} MiB")
                if dw is not None:
                    dc2.metric("Disk Write (cumul.)", f"{dw:.1f} MiB")


# ---------------------------------------------------------------------------
# Memory / RAG section
# ---------------------------------------------------------------------------

_MEM_EVENT_ICON: dict[str, str] = {
    "store_profile": "📊",
    "store_insights": "💡",
    "store_report": "📄",
    "retrieve_context": "🔍",
}
_MEM_STATUS_STYLE: dict[str, str] = {
    "success": "background:#d1fae5;color:#065f46;border:1px solid #059669;",
    "failed": "background:#fee2e2;color:#991b1b;border:1px solid #dc2626;",
    "skipped": "background:#fef3c7;color:#92400e;border:1px solid #d97706;",
    "empty": "background:#f1f5f9;color:#475569;border:1px solid #94a3b8;",
}
_MEM_STATUS_BADGE: dict[str, str] = {
    "success": "✅",
    "failed": "❌",
    "skipped": "⏭️",
    "empty": "⬜",
}


def _render_memory_section(
    dataset_id: str | None,
    rag_stored: bool,
    rag_summary: str | None,
    memory_trace: list[dict[str, Any]],
) -> None:
    """Render the ChromaDB memory section in the diagnostics tab."""
    st.markdown("### 🧠 Memory & RAG")

    # Header row: dataset_id + overall stored status
    h1, h2 = st.columns([3, 1])
    with h1:
        if dataset_id:
            st.caption(f"**Dataset ID:** `{dataset_id}`")
        else:
            st.caption("**Dataset ID:** —")
    with h2:
        if rag_stored:
            st.success("✅ Stored")
        else:
            st.info("⬜ Not stored")

    if rag_summary:
        st.caption(rag_summary)

    if not memory_trace:
        st.caption("No memory events recorded for this run.")
        return

    # Summary counts
    success_count = sum(1 for e in memory_trace if e.get("status") == "success")
    failed_count = sum(1 for e in memory_trace if e.get("status") == "failed")
    total = len(memory_trace)
    if failed_count:
        st.caption(f"{success_count}/{total} events succeeded · {failed_count} failed")
    else:
        st.caption(f"{total} event(s) — all successful")

    # Render each memory event as a compact pill row
    for evt in memory_trace:
        event_key = evt.get("event", "unknown")
        status = evt.get("status", "unknown")
        icon = _MEM_EVENT_ICON.get(event_key, "📦")
        badge = _MEM_STATUS_BADGE.get(status, "❓")
        style = _MEM_STATUS_STYLE.get(status, "")
        chunks = evt.get("chunks", 0)
        message = evt.get("message", "")
        collection = evt.get("collection", "")

        label = event_key.replace("_", " ").capitalize()
        chunks_str = f" · {chunks} chunks" if chunks else ""
        col_str = f" → `{collection}`" if collection else ""

        st.markdown(
            f'<div style="padding:6px 12px;border-radius:8px;margin:3px 0;'
            f'font-family:sans-serif;font-size:13px;{style}">'
            f"<b>{icon} {label}</b> {badge}{col_str}{chunks_str}"
            + (f"<br><span style='font-size:11px;opacity:0.8;'>{message}</span>" if message else "")
            + "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_diagnostics_tab(result: dict[str, Any], key_suffix: str = "") -> None:
    """Render the full diagnostics tab.

    Called from ``app/main.py`` — this is the **only** public function the
    UI layer needs.
    """
    trace: list[dict[str, Any]] = result.get("graph_trace", [])

    if not trace:
        st.info("No graph trace available.")
        return

    # 1️⃣ Aggregated summary
    _render_aggregated_summary(trace)

    st.divider()

    # 2️⃣ Per-node details — deduplicated by node name, keep latest
    st.markdown("### 🔍 Per-Agent Diagnostics")
    seen: dict[str, dict[str, Any]] = {}
    for entry in trace:
        node = entry.get("node", "?")
        seen[node] = entry  # last entry wins → no duplicates

    for node in _NODE_ORDER:
        if node in seen:
            _render_node_card(seen[node])

    # Render any unexpected nodes not in _NODE_ORDER
    for node, entry in seen.items():
        if node not in _NODE_ORDER:
            _render_node_card(entry)

    # 3️⃣ Memory / RAG section
    memory_trace: list[dict[str, Any]] = result.get("memory_trace") or []
    dataset_id: str | None = result.get("dataset_id")
    rag_stored: bool = bool(result.get("rag_stored"))
    rag_summary: str | None = result.get("rag_summary")

    if dataset_id or memory_trace:
        st.divider()
        _render_memory_section(
            dataset_id=dataset_id,
            rag_stored=rag_stored,
            rag_summary=rag_summary,
            memory_trace=memory_trace,
        )

    # 4️⃣ Global error
    global_error = result.get("error")
    if global_error:
        st.error(f"🚨 Pipeline error: {global_error}")

    # 5️⃣ Download full trace
    st.divider()
    st.download_button(
        label="⬇️ Download Full Trace (.json)",
        data=json.dumps(trace, indent=2, ensure_ascii=False, default=str),
        file_name="pipeline_trace.json",
        mime="application/json",
        key=f"dl_trace{key_suffix}",
    )
