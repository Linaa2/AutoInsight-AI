"""app/uncertainty_view.py — Reusable Streamlit renderer for uncertainty scores.

Single public function:
    render_confidence_scores(confidence_scores, *, key_suffix, show_header)

Intentionally self-contained — no dependency on the pipeline or any agent.
Future additions (critic details, P7 judge scores) can be added here without
touching main.py.

Usage::

    from app.uncertainty_view import render_confidence_scores

    render_confidence_scores(result.get("confidence_scores"), key_suffix="_tab")
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEVEL_CONFIG: dict[str, dict[str, str]] = {
    "high": {"icon": "🟢", "label": "High", "color": "#065f46", "bg": "#d1fae5"},
    "medium": {"icon": "🟡", "label": "Medium", "color": "#92400e", "bg": "#fef3c7"},
    "low": {"icon": "🔴", "label": "Low", "color": "#991b1b", "bg": "#fee2e2"},
}

_DRIVER_LABELS: dict[str, str] = {
    "data_quality": "📦 Data Quality",
    "specificity": "🔬 Specificity",
    "statistical_evidence": "📐 Statistical Evidence",
    "critic_assessment": "🧐 Critic Assessment",
}

# Placeholder shown for drivers that are not yet available (e.g. critic not run)
_DRIVER_NA_REASON = "Not yet available — pending agent"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _badge_html(level: str, score: int) -> str:
    """Return an inline HTML confidence badge."""
    cfg = _LEVEL_CONFIG.get(level, _LEVEL_CONFIG["medium"])
    return (
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};'
        f"font-weight:700;font-size:0.82em;padding:3px 10px;"
        f'border-radius:12px;white-space:nowrap;">'
        f"{cfg['icon']} {cfg['label']} &nbsp;{score}%</span>"
    )


def _driver_row(label: str, score: int, max_score: int, reason: str) -> None:
    """Render a single driver as a labelled progress bar + reason caption."""
    col_label, col_bar, col_score = st.columns([3, 5, 1])
    with col_label:
        st.caption(label)
    with col_bar:
        st.progress(score / max_score if max_score else 0)
    with col_score:
        st.caption(f"**{score}/{max_score}**")
    st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;_{reason}_")


def _kpi_summary(scores: list[dict[str, Any]]) -> None:
    """Render a 4-column KPI row: total + high / medium / low counts."""
    high = sum(1 for s in scores if s.get("confidence_level") == "high")
    medium = sum(1 for s in scores if s.get("confidence_level") == "medium")
    low = sum(1 for s in scores if s.get("confidence_level") == "low")

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Insights Scored", len(scores))
    c1.metric("🟢 High", high)
    c2.metric("🟡 Medium", medium)
    c3.metric("🔴 Low", low)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_confidence_scores(
    confidence_scores: list[dict[str, Any]] | None,
    *,
    key_suffix: str = "",
    show_header: bool = True,
) -> None:
    """Render uncertainty estimator results in the current Streamlit context.

    Designed to be called from any tab or widget — self-contained, no side effects.

    Args:
        confidence_scores:
            List of confidence score dicts produced by ``UncertaintyEstimator``.
            Each dict has the schema::

                {
                    "insight_title":    str,
                    "confidence_score": int,      # 0-100
                    "confidence_level": str,      # "high" | "medium" | "low"
                    "drivers": {
                        "data_quality":         {"score": int, "max": int, "reason": str},
                        "specificity":          {"score": int, "max": int, "reason": str},
                        "statistical_evidence": {"score": int, "max": int, "reason": str},
                        "critic_assessment":    {"score": int, "max": int, "reason": str},
                    },
                    "summary": str,
                }

            Pass ``None`` or an empty list to render a "not yet available" notice.

        key_suffix:
            Appended to every Streamlit widget key to avoid duplicate-key errors
            when the function is called from multiple tabs simultaneously.

        show_header:
            When ``True`` (default) a section header + KPI row are rendered.
            Pass ``False`` when embedding inline inside an insight card that
            already has its own header.

    Extension points (for future agents):
        - Critic details: add a ``critiques: list[dict] | None = None`` param
          and render a "Critic Notes" section per insight when provided.
        - P7 judge scores: add a ``judge_scores: list[dict] | None = None`` param
          and show a combined trustworthiness radar.
    """
    if not confidence_scores:
        st.info(
            "🎯 Confidence scores are not yet available. "
            "They are computed automatically as part of the analysis pipeline."
        )
        return

    if show_header:
        st.markdown("### 🎯 Confidence Scores")
        st.caption(
            "Each insight is scored on four drivers (0-25 pts each) using "
            "deterministic rules and a targeted LLM assessment."
        )
        _kpi_summary(confidence_scores)
        st.divider()

    for i, score_dict in enumerate(confidence_scores):
        title: str = score_dict.get("insight_title", f"Insight {i + 1}")
        total: int = int(score_dict.get("confidence_score", 0))
        level: str = str(score_dict.get("confidence_level", "medium"))
        summary: str = str(score_dict.get("summary", ""))
        drivers: dict[str, Any] = score_dict.get("drivers", {})

        cfg = _LEVEL_CONFIG.get(level, _LEVEL_CONFIG["medium"])
        expander_label = f"{cfg['icon']} **{total}%** — {title}"

        with st.expander(expander_label, expanded=(level == "low"), key=f"uc_{i}{key_suffix}"):
            # Badge + summary line
            st.markdown(
                _badge_html(level, total) + "&nbsp;&nbsp;" + summary,
                unsafe_allow_html=True,
            )
            st.divider()

            # Driver breakdown
            st.markdown("**Score breakdown**")
            for driver_key, driver_label in _DRIVER_LABELS.items():
                driver_data: dict[str, Any] = drivers.get(driver_key, {})
                d_score: int = int(driver_data.get("score", 0))
                d_max: int = int(driver_data.get("max", 25))
                d_reason: str = str(driver_data.get("reason", _DRIVER_NA_REASON))
                _driver_row(driver_label, d_score, d_max, d_reason)

        # Small gap between cards
        st.write("")
