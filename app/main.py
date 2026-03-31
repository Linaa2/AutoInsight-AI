"""AutoInsight AI — Unified Streamlit application.

Progressive multi-agent analysis: each agent result is displayed as soon as
it becomes available, giving the user immediate feedback.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import streamlit as st
from streamlit_lottie import st_lottie

from app.langfuse_view import build_langfuse_run_summary
from app.memory_view import (
    CATEGORY_ICONS,
    INSIGHT_CATEGORIES,
    build_memory_status,
    retrieve_analysis_context,
    retrieve_high_priority_insights,
    retrieve_insights_by_category,
)
from config.settings import REPO_ROOT, settings
from diagnostics.renderer import render_diagnostics_tab, render_pipeline_diagram
from orchestration.graph import _AGENTS_ORDER, stream_analysis
from tools.data_loader import DataLoader, UnsupportedFormatError
from tools.profiler_engine import DataProfiler
from utils.langfuse_client import is_langfuse_enabled
from utils.memory import ContextStore
from visualization.executor import execute_chart

if TYPE_CHECKING:
    import pandas as pd

    from tools.profiler_engine import DataProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ASSETS_DIR = REPO_ROOT / "assets"
_LOGO_PATH = _ASSETS_DIR / "auto_insight_ai_logo.png"
_LOTTIE_PATH = _ASSETS_DIR / "ai_agent_animation.json"
_BROKEN_AGENT_PATH = _ASSETS_DIR / "ai_agent_broken.png"

_AGENT_META: dict[str, dict[str, str]] = {
    "profiler": {"icon": "📊", "label": "Profiler", "verb": "Profiling dataset…"},
    "analyst": {"icon": "💡", "label": "Analyst", "verb": "Generating insights…"},
    "visualizer": {"icon": "📈", "label": "Visualizer", "verb": "Creating charts…"},
    "reporter": {"icon": "📄", "label": "Reporter", "verb": "Writing report…"},
}

_STATUS_BADGE = {"success": "✅", "failed": "❌", "skipped": "⏭️"}
_PRIORITY_ICONS = {"high": "🔴", "medium": "🟡", "low": "🟢"}
_CATEGORY_ICONS = {
    "trend": "📈",
    "anomaly": "⚠️",
    "correlation": "🔗",
    "distribution": "📊",
    "general": "💡",
}


@st.cache_data
def _load_lottie() -> dict[str, Any] | None:
    """Load the Lottie animation JSON (cached)."""
    if _LOTTIE_PATH.exists():
        result = json.loads(_LOTTIE_PATH.read_text())
        return result if isinstance(result, dict) else None
    return None


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
/* Smooth card containers */
div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.2);
    margin-bottom: 0.5rem;
}
/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.08) 100%);
    border-radius: 12px;
    padding: 12px 16px;
    border: 1px solid rgba(128,128,128,0.15);
}
/* Prevent metric text truncation */
div[data-testid="stMetricValue"] {
    overflow: visible !important;
    white-space: normal !important;
    word-break: break-word;
    font-size: clamp(0.85rem, 1.5vw, 1.25rem) !important;
}
div[data-testid="stMetricLabel"] {
    white-space: normal !important;
    font-size: 0.78rem !important;
}
/* Tab styling */
button[data-baseweb="tab"] {
    font-weight: 600;
}
/* Agent status pills */
.agent-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 2px 4px;
}
.pill-success { background: #d1fae5; color: #065f46; }
.pill-failed  { background: #fee2e2; color: #991b1b; }
.pill-skipped { background: #fef3c7; color: #92400e; }
.pill-running { background: #dbeafe; color: #1e40af; }
</style>
"""


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

_STATE_KEYS = ("analysis_result", "last_file_name", "analysis_running", "analysis_cancelled")


def _reset_analysis() -> None:
    for key in _STATE_KEYS:
        st.session_state.pop(key, None)


def _cancel_analysis() -> None:
    """``on_click`` callback for the Stop button."""
    st.session_state["analysis_cancelled"] = True
    st.session_state["analysis_running"] = False


def _merge_state(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Merge a node update into the cumulative state, appending graph_trace."""
    merged = dict(base)
    for key, value in update.items():
        if key == "graph_trace" and key in merged:
            merged[key] = list(merged[key]) + list(value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Live pipeline status (HTML only — zero Streamlit components, zero reruns)
# ---------------------------------------------------------------------------


def _render_html_pipeline_status(
    completed: list[str],
    current_agent: str,
    trace: list[dict[str, Any]],
) -> None:
    """Render the pipeline status as plain HTML during analysis streaming.

    Uses only ``st.markdown`` so it never mounts a Streamlit custom component
    and therefore never triggers a script rerun that would kill the streaming
    generator inside ``_run_progressive_analysis``.
    """
    dur_map = {e["node"]: e.get("duration_s", 0.0) for e in trace if "node" in e}
    status_map = {e["node"]: e.get("status", "pending") for e in trace if "node" in e}

    _STYLE: dict[str, tuple[str, str, str, str]] = {
        #             bg        border    color     border-style
        "success": ("d1fae5", "059669", "065f46", "solid"),
        "failed": ("fee2e2", "dc2626", "991b1b", "solid"),
        "skipped": ("fef3c7", "d97706", "92400e", "solid"),
        "running": ("dbeafe", "3b82f6", "1e40af", "solid"),
        "pending": ("f1f5f9", "94a3b8", "475569", "dashed"),
    }
    _BADGE = {"success": "✅", "failed": "❌", "skipped": "⏭️", "running": "⏳", "pending": "⬜"}

    parts: list[str] = []

    # START sentinel
    parts.append(
        '<div style="background:#e0e7ff;border:2px solid #4338ca;color:#3730a3;'
        "border-radius:20px;padding:8px 16px;font-weight:700;font-size:12px;"
        "font-family:sans-serif;white-space:nowrap;"
        'box-shadow:0 2px 6px rgba(0,0,0,0.10);">&#9654; START</div>'
    )

    prev_done = True  # START is always considered "done"
    for agent_id in _AGENTS_ORDER:
        meta = _AGENT_META[agent_id]
        status = status_map.get(agent_id, "pending")
        if agent_id == current_agent and status == "pending":
            status = "running"

        bg, border, color, bstyle = _STYLE.get(status, _STYLE["pending"])
        badge = _BADGE.get(status, "⬜")
        dur = dur_map.get(agent_id, 0.0)
        dur_str = f"{dur:.1f}s" if status not in {"pending", "running"} else "\u2026"

        # Edge arrow between nodes
        arrow_col = f"#{border}" if prev_done else "#94a3b8"
        parts.append(
            f'<div style="color:{arrow_col};font-size:16px;flex-shrink:0;'
            f'padding:0 4px;opacity:{1.0 if prev_done else 0.4};">&#8212;&#9654;</div>'
        )

        parts.append(
            f'<div style="background:#{bg};border:2px {bstyle} #{border};color:#{color};'
            f"border-radius:10px;padding:10px 14px;min-width:110px;text-align:center;"
            f"font-family:sans-serif;font-size:13px;flex-shrink:0;"
            f'box-shadow:0 2px 6px rgba(0,0,0,0.10);">'
            f'<div style="font-weight:700;">{meta["icon"]} {meta["label"]}</div>'
            f'<div style="font-size:11px;margin-top:4px;">{badge} {status.capitalize()} · {dur_str}</div>'
            f"</div>"
        )
        prev_done = status == "success"

    # END sentinel edge + node
    end_col = "#059669" if "reporter" in completed else "#94a3b8"
    parts.append(
        f'<div style="color:{end_col};font-size:16px;flex-shrink:0;'
        f'padding:0 4px;opacity:{1.0 if "reporter" in completed else 0.4};">&#8212;&#9654;</div>'
    )
    parts.append(
        '<div style="background:#e0e7ff;border:2px solid #4338ca;color:#3730a3;'
        "border-radius:20px;padding:8px 16px;font-weight:700;font-size:12px;"
        "font-family:sans-serif;white-space:nowrap;"
        'box-shadow:0 2px 6px rgba(0,0,0,0.10);">&#9632; END</div>'
    )

    html = (
        '<div style="display:flex;align-items:center;justify-content:center;'
        "flex-wrap:nowrap;gap:2px;padding:18px 12px;"
        "background:rgba(99,102,241,0.04);border-radius:12px;"
        'border:1px solid rgba(99,102,241,0.15);overflow-x:auto;">' + "".join(parts) + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_landing() -> None:
    """Initial landing content before file upload."""
    st.markdown(
        """
        ### Welcome to AutoInsight AI 👋

        Upload a **CSV**, **Excel**, or **Parquet** file and let our
        multi-agent team do the heavy lifting:

        | Agent | What it does |
        |-------|-------------|
        | 📊 **Profiler** | Deterministic statistics + LLM interpretation |
        | 💡 **Analyst** | Actionable business insights |
        | 📈 **Visualizer** | Auto-generated interactive charts |
        | 📄 **Reporter** | Executive summary report |

        Results appear **progressively** — no need to wait for everything!
        """
    )


def _render_kpi_row(profile: DataProfile) -> None:
    cols = st.columns(5)
    cols[0].metric("Rows", f"{profile.shape[0]:,}")
    cols[1].metric("Columns", f"{profile.shape[1]:,}")
    cols[2].metric("Duplicates", f"{profile.duplicates_count:,}")
    cols[3].metric("Missing", f"{profile.total_missing_pct:.1f}%")
    cols[4].metric("Memory", f"{profile.memory_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_profile_tab(result: dict[str, Any], key_suffix: str = "") -> None:
    md: str | None = result.get("profile_markdown")
    if not md:
        st.info("Profile not available — the profiler agent may have failed.")
        return

    # Summary header
    trace = result.get("graph_trace", [])
    profiler_trace = next((t for t in trace if t.get("node") == "profiler"), None)
    if profiler_trace:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.success(f"✅ {profiler_trace.get('summary', 'Profile generated')}")
        with c2:
            dur = profiler_trace.get("duration_s", 0)
            st.caption(f"⏱️ {dur:.1f}s")

    st.divider()
    st.markdown(md)

    st.divider()
    st.download_button(
        label="⬇️ Download Profile (.md)",
        data=md,
        file_name="dataset_profile.md",
        mime="text/markdown",
        key=f"dl_profile{key_suffix}",
    )


def _render_insights_tab(result: dict[str, Any], key_suffix: str = "") -> None:
    insights: list[dict[str, Any]] | None = result.get("insights")
    md: str | None = result.get("insights_markdown")

    if not insights and not md:
        st.info("No insights available — the analyst agent may have been skipped.")
        return

    # Summary header
    trace = result.get("graph_trace", [])
    analyst_trace = next((t for t in trace if t.get("node") == "analyst"), None)
    if analyst_trace:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.success(f"✅ {analyst_trace.get('summary', 'Insights generated')}")
        with c2:
            dur = analyst_trace.get("duration_s", 0)
            st.caption(f"⏱️ {dur:.1f}s")

    if insights:
        # Priority summary
        high = sum(1 for i in insights if i.get("priority") == "high")
        med = sum(1 for i in insights if i.get("priority") == "medium")
        low = sum(1 for i in insights if i.get("priority") == "low")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Insights", len(insights))
        sc2.metric("🔴 High Priority", high)
        sc3.metric("🟡 Medium", med)
        sc4.metric("🟢 Low", low)

    st.divider()

    view = st.radio(
        "Display mode", ["Cards", "Markdown"], horizontal=True, key=f"insight_view{key_suffix}"
    )

    if view == "Markdown" and md:
        st.markdown(md)
    elif insights:
        for ins in insights:
            priority = ins.get("priority", "medium")
            category = ins.get("category", "general")
            p_icon = _PRIORITY_ICONS.get(priority, "⚪")
            c_icon = _CATEGORY_ICONS.get(category, "💡")
            title = ins.get("title", "Insight")

            with st.expander(f"{p_icon} {c_icon} {title}", expanded=True):
                st.markdown(
                    f"**Category:** {c_icon} {category.capitalize()} &nbsp;|&nbsp; "
                    f"**Priority:** {p_icon} {priority.capitalize()}"
                )
                st.divider()

                col_obs, col_hyp = st.columns(2)
                with col_obs:
                    st.markdown("**🔎 Observation**")
                    st.info(ins.get("observation", "—"))
                with col_hyp:
                    st.markdown("**🧪 Hypothesis**")
                    st.warning(ins.get("hypothesis", "—"))

                st.markdown("**✅ Recommendation**")
                st.success(ins.get("recommendation", "—"))
    else:
        st.markdown(md or "")

    st.divider()
    if md:
        st.download_button(
            label="⬇️ Download Insights (.md)",
            data=md,
            file_name="insights.md",
            mime="text/markdown",
            key=f"dl_insights{key_suffix}",
        )
    if insights:
        st.download_button(
            label="⬇️ Download Insights (.json)",
            data=json.dumps(insights, indent=2, ensure_ascii=False),
            file_name="insights.json",
            mime="application/json",
            key=f"dl_insights_json{key_suffix}",
        )


def _render_visualizations_tab(
    result: dict[str, Any], df: pd.DataFrame, key_suffix: str = ""
) -> None:
    viz: dict[str, Any] | None = result.get("visualization_result")
    if not viz:
        st.info("No visualizations available — the visualizer agent may have been skipped.")
        return

    charts: list[dict[str, Any]] = viz.get("charts", [])
    if not charts:
        parsing_err = viz.get("parsing_error")
        if parsing_err:
            st.warning(f"Chart parsing error: {parsing_err}")
        else:
            st.info("The visualizer did not produce any charts.")
        return

    # Summary header
    n_ok = sum(1 for c in charts if c.get("execution", {}).get("success"))
    trace = result.get("graph_trace", [])
    viz_trace = next((t for t in trace if t.get("node") == "visualizer"), None)
    if viz_trace:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.success(f"✅ {n_ok}/{len(charts)} charts rendered successfully")
        with c2:
            dur = viz_trace.get("duration_s", 0)
            st.caption(f"⏱️ {dur:.1f}s")

    st.divider()

    for i, chart_data in enumerate(charts, 1):
        spec_dict: dict[str, Any] = chart_data.get("spec", {})
        exec_data: dict[str, Any] = chart_data.get("execution", {})
        title = spec_dict.get("title", f"Chart {i}")
        chart_type = spec_dict.get("chart_type", "")

        with st.container():
            head_col, badge_col = st.columns([4, 1])
            with head_col:
                st.markdown(f"### {i}. {title}")
            with badge_col:
                st.code(chart_type, language=None)

            explanation = spec_dict.get("explanation")
            if explanation:
                st.caption(f"💬 {explanation}")

            if exec_data.get("success"):
                code = spec_dict.get("code", "")
                if code:
                    exec_result = execute_chart(df, code)
                    if exec_result.success and exec_result.figure is not None:
                        st.plotly_chart(
                            exec_result.figure,
                            use_container_width=True,
                            key=f"chart_{i}{key_suffix}",
                        )
                    else:
                        st.error(f"Re-execution failed: {exec_result.error}")
                else:
                    st.warning("No code available for this chart.")
            else:
                st.error(f"Execution error: {exec_data.get('error', 'Unknown')}")

            with st.expander("🔧 View generated code", expanded=False):
                st.code(spec_dict.get("code", ""), language="python")

            st.divider()


def _render_report_tab(result: dict[str, Any], key_suffix: str = "") -> None:
    report: str | None = result.get("report_markdown")
    if not report:
        st.info("No report available — the reporter agent may have been skipped.")
        return

    # Memory enrichment banner
    if result.get("rag_analysis_context"):
        st.info(
            "📚 This report was **enriched with context** retrieved from previous "
            "analyses stored in memory. Switch to the **🧠 Memory** tab to explore."
        )

    # Summary header
    trace = result.get("graph_trace", [])
    reporter_trace = next((t for t in trace if t.get("node") == "reporter"), None)
    if reporter_trace:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.success(f"✅ Report generated ({len(report):,} characters)")
        with c2:
            dur = reporter_trace.get("duration_s", 0)
            st.caption(f"⏱️ {dur:.1f}s")

    st.divider()
    st.markdown(report)

    st.divider()
    col_md, col_txt = st.columns(2)
    with col_md:
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=report,
            file_name="analysis_report.md",
            mime="text/markdown",
            key=f"dl_report_md{key_suffix}",
        )
    with col_txt:
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=report,
            file_name="analysis_report.txt",
            mime="text/plain",
            key=f"dl_report_txt{key_suffix}",
        )


# _render_diagnostics replaced by diagnostics.renderer.render_diagnostics_tab


# ---------------------------------------------------------------------------
# Memory tab
# ---------------------------------------------------------------------------


def _render_memory_tab(result: dict[str, Any], key_suffix: str = "") -> None:
    """Render the dedicated Memory / RAG showcase tab."""
    status = build_memory_status(result)

    # ── Section 1: Memory Status Overview ──────────────────────────────────
    st.markdown("### 📋 Memory Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", status.dataset_id or "—")
    c2.metric("Memory", "Available" if status.memory_available else "Unavailable")
    c3.metric(
        "Prior Context",
        f"{status.previous_context_chars:,} chars" if status.previous_context_found else "None",
    )
    c4.metric("Stored", "Yes" if status.current_run_stored else "No")

    # ── Section 2: Retrieved Context Preview ───────────────────────────────
    rag_ctx = result.get("rag_analysis_context") or ""
    if rag_ctx:
        st.divider()
        st.markdown("### 🔍 Retrieved Context (fed to Reporter)")
        st.info(
            "The following context was automatically retrieved from memory and "
            "injected into the Reporter agent to enrich the final report."
        )
        with st.expander("View retrieved context", expanded=False):
            st.text(rag_ctx)
    else:
        st.divider()
        st.markdown("### 🔍 Retrieved Context")
        st.caption(
            "No prior context was found for this dataset. "
            "Run a second analysis on the same file to see memory retrieval in action."
        )

    # ── Section 3: Stored Artifact Summary ─────────────────────────────────
    st.divider()
    st.markdown("### 📦 Stored Artifacts")
    if status.stored_artifacts:
        artifact_icons = {"profile": "📊", "insights": "💡", "report": "📄"}
        cols = st.columns(len(status.stored_artifacts))
        for col, art in zip(cols, status.stored_artifacts):
            col.success(f"{artifact_icons.get(art, '📦')} {art.capitalize()}")
        st.caption(
            f"{status.total_chunks_stored} chunk(s) stored across "
            f"{len(status.collections_used)} collection(s)"
        )
        if status.rag_summary:
            st.caption(status.rag_summary)
    else:
        st.caption("No artifacts were stored during this run.")

    # ── Section 4: Interactive Memory Retrieval ────────────────────────────
    dataset_id = status.dataset_id
    if dataset_id and status.memory_available:
        st.divider()
        st.markdown("### 🧪 Memory Retrieval Demo")
        st.caption(
            "Query the ChromaDB memory live. These are the same functions that "
            "future agents (e.g. Text-to-Code) will use to ground their work."
        )

        demo_tab1, demo_tab2, demo_tab3 = st.tabs(
            ["🔝 High Priority", "🏷️ By Category", "📑 Full Context"]
        )

        with demo_tab1:
            if st.button("Retrieve high-priority insights", key=f"mem_hp{key_suffix}"):
                with st.spinner("Querying memory…"):
                    res = retrieve_high_priority_insights(dataset_id)
                if res.empty:
                    st.warning("No high-priority insights found in memory yet.")
                else:
                    st.text(res.content)

        with demo_tab2:
            selected = st.selectbox(
                "Category",
                INSIGHT_CATEGORIES,
                format_func=lambda c: f"{CATEGORY_ICONS.get(c, '')} {c.capitalize()}",
                key=f"mem_cat_sel{key_suffix}",
            )
            if st.button("Retrieve by category", key=f"mem_cat_btn{key_suffix}"):
                with st.spinner("Querying memory…"):
                    res = retrieve_insights_by_category(dataset_id, selected)
                if res.empty:
                    st.warning(f"No {selected} insights found in memory yet.")
                else:
                    st.text(res.content)

        with demo_tab3:
            if st.button("Retrieve full analysis context", key=f"mem_full{key_suffix}"):
                with st.spinner("Querying memory…"):
                    res = retrieve_analysis_context(dataset_id)
                if res.empty:
                    st.warning("No analysis context found in memory yet.")
                else:
                    st.text(res.content)

    # ── Section 5: Why Memory Matters ──────────────────────────────────────
    st.divider()
    with st.expander("\u2139\ufe0f Why Memory Matters", expanded=False):
        st.markdown(
            """
**AutoInsight-AI stores every analysis in a ChromaDB vector database.**

On subsequent runs, the system retrieves relevant context from past analyses
and feeds it to downstream agents. This enables:

- **Richer reports** — the Reporter sees trends that span multiple runs.
- **Grounded code generation** — the future Text-to-Code agent will use
  retrieved insights to write more accurate data-science code.
- **Conversational follow-ups** — ask questions and get answers that
  reference your full analytical history.

**Try it:** run the same dataset twice and compare the Report tab — the
second run will show an "enriched with memory" banner.
"""
        )


# ---------------------------------------------------------------------------
# Observability tab
# ---------------------------------------------------------------------------

_OBS_NODE_ORDER = ["profiler", "analyst", "visualizer", "reporter", "rag_storage"]
_OBS_STATUS_EMOJI: dict[str, str] = {
    "success": "✅",
    "failed": "❌",
    "skipped": "⏭️",
    "unknown": "❓",
}


def _render_observability_tab(result: dict[str, Any]) -> None:
    """Render the 🔭 Observability tab — LangFuse showcase panel."""
    summary = build_langfuse_run_summary(result)

    # ── 1. LangFuse Status Card ───────────────────────────────────────────
    st.markdown("### 🔭 LangFuse Observability Status")
    if not summary.enabled:
        st.warning(
            "**LangFuse is disabled.** Set `LANGFUSE_ENABLED=true` and provide "
            "`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in your `.env` to enable "
            "full GenAI tracing. Run `make langfuse-up` to start a local instance."
        )
        st.caption("The analysis ran successfully — LangFuse adds *external* observability on top.")
    elif summary.trace_id:
        st.success(
            f"✅ **LangFuse active** — host: [{summary.host}]({summary.host}) | Trace captured ✓"
        )
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Host", summary.host or "—")
        col_b.metric("Trace ID", f"`{summary.trace_id}`")
        col_c.metric("Dataset", summary.dataset_id or "—")
    else:
        st.error(
            "LangFuse is enabled but no trace ID was captured for this run. "
            "Check that the LangFuse server is reachable at "
            f"[{summary.host}]({summary.host}) and restart the analysis."
        )

    # ── 2. Run Summary ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Run Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Status", summary.status.capitalize())
    m2.metric("Duration", f"{summary.duration_s:.1f}s" if summary.duration_s else "—")
    m3.metric("✅ Succeeded", summary.nodes_succeeded)
    m4.metric("❌ Failed", summary.nodes_failed)
    m5.metric("⏭️ Skipped", summary.nodes_skipped)

    if summary.started_at:
        st.caption(f"Run started: `{summary.started_at}`")
    if summary.dataset_id:
        st.caption(f"Dataset: `{summary.dataset_id}` · File: `{summary.file_name or '—'}`")

    # ── 3. Open in LangFuse ───────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔗 Open in LangFuse")

    if summary.trace_url:
        st.link_button(
            "🚀 Open this run in LangFuse",
            summary.trace_url,
            use_container_width=True,
        )
        st.caption(
            "This link opens the **complete trace** in LangFuse — all agent spans, "
            "prompt texts, LLM responses, latency data, and token counts are there. "
            "Click any span to inspect the exact prompt that was sent to the model."
        )
    elif summary.trace_id:
        st.code(summary.trace_id, language=None)
        st.caption(
            "Copy this trace ID and search for it in LangFuse "
            f"([{summary.host}]({summary.host})) to inspect the full trace."
        )
    elif summary.enabled:
        st.info("No trace ID captured for this run — start a new analysis to generate one.")
    else:
        with st.expander("\u2139\ufe0f How to enable LangFuse", expanded=False):
            st.markdown(
                """
1. Start a local LangFuse instance: `make langfuse-up`
2. Open the LangFuse UI at **http://localhost:3001**
3. Set `LANGFUSE_ENABLED=true` in your `.env`
4. Re-run the analysis — a clickable trace link will appear here.
"""
            )

    # ── 4. Node-Level Execution Table ─────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ Node Execution Summary")

    if summary.node_summaries:
        # Column headers
        hcols = st.columns([2, 1, 1, 3, 3])
        hcols[0].markdown("**Node**")
        hcols[1].markdown("**Status**")
        hcols[2].markdown("**Duration**")
        hcols[3].markdown("**Model**")
        hcols[4].markdown("**Outcome**")
        st.markdown("---")

        ordered = sorted(
            summary.node_summaries,
            key=lambda n: (
                _OBS_NODE_ORDER.index(n.node_name) if n.node_name in _OBS_NODE_ORDER else 99
            ),
        )
        for node in ordered:
            badge = _OBS_STATUS_EMOJI.get(node.status, "❓")
            dur_str = f"{node.duration_s:.1f}s" if node.duration_s is not None else "—"
            if node.model_name:
                model_str = f"`{node.model_name}`"
                if node.provider:
                    model_str += f" ({node.provider})"
            else:
                model_str = "—"

            row = st.columns([2, 1, 1, 3, 3])
            row[0].markdown(f"**{node.node_name}**")
            row[1].markdown(badge)
            row[2].markdown(dur_str)
            row[3].markdown(model_str)
            row[4].markdown(node.summary or "—")

            if node.error:
                st.error(f"**{node.node_name}** failed: {node.error}")

        st.caption(
            "Each row above corresponds to a **LangFuse span** inside the trace. "
            "Open the trace link to inspect prompts, model responses, and token usage."
        )
    else:
        st.caption("No node execution data available.")

    # ── 5. RAG / Memory Observability ─────────────────────────────────────
    st.divider()
    st.markdown("### 🧠 RAG / Memory Observability")

    rag1, rag2, rag3 = st.columns(3)
    rag1.metric("Memory Events", summary.memory_events_count)
    rag2.metric("Stored to DB", "Yes ✅" if summary.memory_stored else "No")
    rag3.metric("Prior Context", "Found ✅" if summary.rag_context_found else "Not found")

    if summary.memory_collections:
        st.caption(
            "Collections written: " + ", ".join(f"`{c}`" for c in summary.memory_collections)
        )
    if summary.memory_events_summary:
        st.caption(summary.memory_events_summary)

    if summary.rag_context_found:
        st.info(
            "📚 Prior analysis context was **retrieved from ChromaDB** and injected into "
            "the Reporter — this retrieval event is logged as a `rag-context-retrieval` span "
            "in LangFuse. Open the trace to verify what context was found and how it "
            "influenced the report."
        )
    else:
        st.caption(
            "💡 No prior context found for this dataset. Run the analysis a **second time** "
            "on the same file to see the full RAG retrieval flow traced in LangFuse."
        )

    # ── 6. Why LangFuse Matters ───────────────────────────────────────────
    st.divider()
    with st.expander("\u2139\ufe0f Why LangFuse Matters for GenAI Systems", expanded=False):
        st.markdown(
            """
**LangFuse provides end-to-end observability that local diagnostics cannot.**

| Capability | Local Diagnostics | LangFuse |
|---|:---:|:---:|
| Per-node timing | ✅ | ✅ |
| Resource usage (CPU/RAM/GPU) | ✅ | — |
| Full prompt text | — | ✅ |
| Full LLM response | — | ✅ |
| Token cost tracking | — | ✅ |
| Multi-run history | — | ✅ |
| Cross-run comparison | — | ✅ |
| RAG retrieval tracing | Partial | ✅ |
| Shareable run links | — | ✅ |

**In a production GenAI system, LangFuse lets you:**
- Debug prompt quality issues without re-running the pipeline
- Identify latency bottlenecks at the model-call level
- Verify that RAG context retrieval is working correctly
- Compare traces before and after a prompt change
- Monitor token costs over time

**In this project:** every analysis run is a single top-level trace.
Each agent node is a child span. RAG retrieval, storage events, and
model calls are all nested inside that trace — making the entire
multi-agent pipeline fully inspectable in one place.
"""
        )

    # ── 7. Raw Identifiers (developer expander) ───────────────────────────
    with st.expander("🛠️ Developer: Raw Run Identifiers", expanded=False):
        id_rows = {
            "trace_id": summary.trace_id or "N/A",
            "dataset_id": summary.dataset_id or "N/A",
            "file_name": summary.file_name or "N/A",
            "langfuse_host": summary.host or "N/A (LangFuse disabled)",
            "trace_url": summary.trace_url or "N/A",
        }
        for k, v in id_rows.items():
            st.code(f"{k}: {v}", language=None)

        if summary.node_summaries:
            st.markdown("**Node statuses:**")
            ordered_dev = sorted(
                summary.node_summaries,
                key=lambda n: (
                    _OBS_NODE_ORDER.index(n.node_name) if n.node_name in _OBS_NODE_ORDER else 99
                ),
            )
            for node in ordered_dev:
                dur_str = f"{node.duration_s:.2f}s" if node.duration_s is not None else "?"
                model_str = node.model_name or "no model"
                st.caption(f"**{node.node_name}**: {node.status} · {dur_str} · {model_str}")


# ---------------------------------------------------------------------------
# Cancelled / result / progressive renderers
# ---------------------------------------------------------------------------


def _render_cancelled() -> None:
    """Show the 'analysis cancelled' screen with broken agent image."""
    st.divider()
    _c1, center, _c3 = st.columns([1, 2, 1])
    with center:
        if _BROKEN_AGENT_PATH.exists():
            st.image(str(_BROKEN_AGENT_PATH), width="content")
        st.markdown(
            "<h3 style='text-align:center;'>Analysis Cancelled</h3>"
            "<p style='text-align:center; color:gray;'>"
            "The analysis was stopped before completion. "
            "Upload a file and run again when you're ready.</p>",
            unsafe_allow_html=True,
        )
    if st.button("🔄 Start Over", use_container_width=True):
        _reset_analysis()
        st.rerun()


def _render_result_tabs(result: dict[str, Any], df: pd.DataFrame) -> None:
    """Render the final completed result tabs."""
    tab_names: list[str] = []
    tab_keys: list[str] = []

    if result.get("profile_markdown"):
        tab_names.append("📊 Profile")
        tab_keys.append("profile")
    if result.get("insights") or result.get("insights_markdown"):
        tab_names.append("💡 Insights")
        tab_keys.append("insights")
    if result.get("visualization_result"):
        tab_names.append("📈 Visualizations")
        tab_keys.append("viz")
    if result.get("report_markdown"):
        tab_names.append("📄 Report")
        tab_keys.append("report")
    # Always offer Memory tab when dataset_id is present
    if result.get("dataset_id"):
        tab_names.append("🧠 Memory")
        tab_keys.append("memory")
    # Observability tab: always show when graph_trace is present (useful
    # regardless of LangFuse enabled state — shows node table + status)
    if result.get("graph_trace") or result.get("langfuse_trace_id"):
        tab_names.append("🔭 Observability")
        tab_keys.append("obs")
    if result.get("graph_trace"):
        tab_names.append("🔧 Diagnostics")
        tab_keys.append("diag")

    if not tab_names:
        st.warning("Analysis completed but no results were produced.")
        if result.get("error"):
            st.error(result["error"])
        return

    tabs = st.tabs(tab_names)
    for tab, key in zip(tabs, tab_keys):
        with tab:
            if key == "profile":
                _render_profile_tab(result)
            elif key == "insights":
                _render_insights_tab(result)
            elif key == "viz":
                _render_visualizations_tab(result, df)
            elif key == "report":
                _render_report_tab(result)
            elif key == "memory":
                _render_memory_tab(result)
            elif key == "obs":
                _render_observability_tab(result)
            elif key == "diag":
                render_diagnostics_tab(result)


def _render_progressive_tabs(
    cumulative: dict[str, Any],
    df: pd.DataFrame,
    completed: list[str],
    next_agent: str,
) -> None:
    """Render tabs during streaming — completed content + placeholders."""
    tab_labels: list[str] = []
    tab_agents: list[str] = []

    for agent in _AGENTS_ORDER:
        meta = _AGENT_META[agent]
        if agent in completed:
            tab_labels.append(f"{meta['icon']} {meta['label']} ✓")
        elif agent == next_agent:
            tab_labels.append(f"{meta['icon']} {meta['label']} ⏳")
        else:
            tab_labels.append(f"{meta['icon']} {meta['label']}")
        tab_agents.append(agent)

    tab_labels.append("🔧 Diagnostics")
    tab_agents.append("diag")

    ks = f"_prog_{len(completed)}"
    tabs = st.tabs(tab_labels)
    for tab, agent in zip(tabs, tab_agents):
        with tab:
            if agent in completed:
                if agent == "profiler":
                    _render_profile_tab(cumulative, key_suffix=ks)
                elif agent == "analyst":
                    _render_insights_tab(cumulative, key_suffix=ks)
                elif agent == "visualizer":
                    _render_visualizations_tab(cumulative, df, key_suffix=ks)
                elif agent == "reporter":
                    _render_report_tab(cumulative, key_suffix=ks)
            elif agent == "diag":
                render_diagnostics_tab(cumulative, key_suffix=ks)
            elif agent == next_agent:
                meta = _AGENT_META[agent]
                st.info(f"{meta['icon']} {meta['label']} agent is running…")
            else:
                st.caption("⏳ Waiting…")


# ---------------------------------------------------------------------------
# LangFuse sidebar status
# ---------------------------------------------------------------------------


def _render_langfuse_sidebar_status(result: dict[str, Any] | None) -> None:
    """Render a compact LangFuse observability status card in the sidebar."""
    enabled = is_langfuse_enabled()
    if not enabled:
        st.caption("🔴 LangFuse: disabled")
        st.caption("Set `LANGFUSE_ENABLED=true` + keys to enable tracing.")
        return

    st.caption(f"🟢 LangFuse: enabled — [{settings.LANGFUSE_HOST}]({settings.LANGFUSE_HOST})")

    if result:
        trace_id = result.get("langfuse_trace_id", "")
        if trace_id:
            trace_url = f"{settings.LANGFUSE_HOST.rstrip('/')}/trace/{trace_id}"
            st.caption(f"[🔍 View trace in LangFuse]({trace_url})")
            st.caption(f"Trace ID: `{trace_id}`")
        else:
            st.caption("Trace ID: not captured")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon="🔍",
        layout="wide",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # ---- Header ----
    _header_left, header_right = st.columns([1, 4])
    with header_right:
        st.title(f"🔍 {settings.APP_TITLE}")
        st.caption(
            "Upload a dataset and let the multi-agent team profile, "
            "analyse, visualise, and report — results appear as each agent finishes."
        )

    # ---- Sidebar ----
    with st.sidebar:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width="content")

        st.header("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Supported: CSV, Excel (.xlsx/.xls), Parquet",
        )
        st.divider()
        st.caption("Built with LangGraph + Streamlit")

        # Memory status — shown after a completed analysis
        result_for_sidebar: dict[str, Any] | None = st.session_state.get("analysis_result")
        if result_for_sidebar and result_for_sidebar.get("rag_stored"):
            st.divider()
            st.success("🧠 Analysis stored in memory")
            did = result_for_sidebar.get("dataset_id", "")
            if did:
                st.caption(f"Dataset: `{did}`")

        # LangFuse observability status
        st.divider()
        _render_langfuse_sidebar_status(result_for_sidebar)

        # Rerun button — only when analysis is complete
        if st.session_state.get("analysis_result") is not None and not st.session_state.get(
            "analysis_running"
        ):
            st.divider()
            if st.button("🔄 Re-run Analysis"):
                _reset_analysis()
                st.rerun()

    if uploaded_file is None:
        _render_landing()
        return

    # ---- Load dataset ----
    loader = DataLoader()
    try:
        file_bytes = uploaded_file.read()
        df = loader.load_from_upload(file_bytes, uploaded_file.name)
    except UnsupportedFormatError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Failed to load file: {exc}")
        return

    # Reset on new file
    if st.session_state.get("last_file_name") != uploaded_file.name:
        _reset_analysis()
        st.session_state["last_file_name"] = uploaded_file.name
        st.session_state["dataset_id"] = ContextStore.make_dataset_id(uploaded_file.name)
    elif "dataset_id" not in st.session_state:
        st.session_state["dataset_id"] = ContextStore.make_dataset_id(uploaded_file.name)

    dataset_id: str = st.session_state["dataset_id"]

    # ---- Dataset preview ----
    profiler_engine = DataProfiler()
    profile = profiler_engine.profile(df)
    _render_kpi_row(profile)

    with st.expander("🗂️ Preview data", expanded=False):
        st.dataframe(df.head(settings.PROFILER_SAMPLE_ROWS), use_container_width=True)

    st.divider()

    # ---- Pipeline Architecture — always visible, updated live during analysis ----
    st.markdown(
        "<h3 style='margin-bottom:4px;'>🔄 Pipeline Status</h3>",
        unsafe_allow_html=True,
    )
    diagram_area = st.empty()

    result: dict[str, Any] | None = st.session_state.get("analysis_result")
    is_cancelled = st.session_state.get("analysis_cancelled", False)

    # Render initial or final diagram state
    _initial_trace: list[dict[str, Any]] = result.get("graph_trace", []) if result else []
    with diagram_area.container():
        render_pipeline_diagram(_initial_trace, key_suffix="_main")

    st.divider()

    # ---- Dispatch based on state ----
    if is_cancelled and result is None:
        _render_cancelled()
        return

    if result is None:
        if st.button(
            "🚀 Run Full Analysis",
            type="primary",
            use_container_width=True,
        ):
            _run_progressive_analysis(df, uploaded_file.name, diagram_area, dataset_id)
            return
        st.info(
            "Click **Run Full Analysis** to start the multi-agent pipeline. "
            "Results will appear progressively as each agent finishes."
        )
        return

    _render_result_tabs(result, df)


# ---------------------------------------------------------------------------
# Progressive analysis runner
# ---------------------------------------------------------------------------


def _run_progressive_analysis(
    df: pd.DataFrame,
    file_name: str,
    diagram_area: Any,
    dataset_id: str = "",
) -> None:
    """Stream the graph execution with progressive tab rendering + Lottie."""
    st.session_state["analysis_running"] = True
    st.session_state.pop("analysis_cancelled", None)

    lottie_data = _load_lottie()
    cumulative: dict[str, Any] = {}
    completed: list[str] = []
    start_time = time.time()

    # ---- Stop button ----
    _sc1, stop_col, _sc3 = st.columns([2, 1, 2])
    with stop_col:
        st.button(
            "🛑 Stop Analysis",
            on_click=_cancel_analysis,
            type="secondary",
            use_container_width=True,
            key="stop_btn",
        )

    # ---- Lottie animation (rendered once, persists throughout) ----
    anim_area = st.empty()
    with anim_area.container():
        _ac1, anim_center, _ac3 = st.columns([1, 5, 1])
        with anim_center:
            if lottie_data:
                st_lottie(lottie_data, height=360, key="running_lottie")

    # ---- Seed diagram using plain HTML (mounts no custom component → no reruns) ----
    with diagram_area.container():
        _render_html_pipeline_status([], "profiler", [])

    # ---- Results area (updated per node) ----
    results_area = st.empty()
    with results_area.container():
        _render_progressive_tabs(cumulative, df, completed, "profiler")

    # ---- Stream nodes ----
    for node_name, node_output in stream_analysis(
        df, file_name=file_name, dataset_id=dataset_id or None
    ):
        if st.session_state.get("analysis_cancelled"):
            break

        completed.append(node_name)
        cumulative = _merge_state(cumulative, dict(node_output))

        idx = _AGENTS_ORDER.index(node_name) if node_name in _AGENTS_ORDER else -1
        next_agent = _AGENTS_ORDER[idx + 1] if idx + 1 < len(_AGENTS_ORDER) else ""

        # Live status via plain HTML — never triggers a Streamlit rerun
        live_trace: list[dict[str, Any]] = cumulative.get("graph_trace", [])
        with diagram_area.container():
            _render_html_pipeline_status(completed, next_agent, live_trace)

        with results_area.container():
            _render_progressive_tabs(cumulative, df, completed, next_agent)

    # ---- Finished ----
    total_time = time.time() - start_time
    anim_area.empty()

    # Store results; st.rerun() causes main() to render the full React Flow diagram
    st.session_state["analysis_result"] = cumulative
    st.session_state["analysis_running"] = False
    st.toast(f"✅ Analysis complete in {total_time:.0f}s!", icon="🎉")
    st.balloons()
    time.sleep(0.5)
    st.rerun()


if __name__ == "__main__":
    main()
