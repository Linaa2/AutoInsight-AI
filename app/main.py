"""AutoInsight AI — Streamlit application entry point (P1: Profiler UI + P2: Analyst UI)."""

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that local modules are importable
# when the app is launched with `streamlit run app/main.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.analyst import AnalystAgent
from agents.profiler import ProfilerAgent
from tools.data_loader import DataLoader, UnsupportedFormatError
from tools.profiler_engine import DataProfile, DataProfiler

load_dotenv()

_APP_TITLE = os.getenv("APP_TITLE", "AutoInsight AI")
_SAMPLE_ROWS = int(os.getenv("PROFILER_SAMPLE_ROWS", "5"))


# ---------------------------------------------------------------------------
# Helpers: bridge DataProfile → analyst input
# ---------------------------------------------------------------------------


def _profile_to_dict(profile: DataProfile) -> dict:
    """Convert a DataProfile object to the dict format expected by AnalystAgent."""
    columns_list = []
    missing_values = {}

    for col_name, cp in profile.columns.items():
        col_info = {
            "name": col_name,
            "dtype": str(cp.dtype),
            "missing": round(cp.missing_pct * profile.shape[0] / 100) if cp.missing_pct else 0,
            "missing_pct": round(cp.missing_pct, 1),
            "unique": cp.unique_count,
        }

        if cp.dtype_category == "numeric":
            col_info["stats"] = {
                "mean": cp.mean,
                "std": cp.std,
                "min": cp.min,
                "25%": cp.q25,
                "50%": cp.median,
                "75%": cp.q75,
                "max": cp.max,
            }
        elif cp.top_values:
            col_info["top_values"] = cp.top_values

        columns_list.append(col_info)

        if cp.missing_pct and cp.missing_pct > 0:
            missing_values[col_name] = col_info["missing"]

    return {
        "shape": {"rows": profile.shape[0], "cols": profile.shape[1]},
        "columns": columns_list,
        "missing_values": missing_values,
        "duplicates": profile.duplicates_count,
        "memory_mb": profile.memory_mb,
    }


def _build_sample_text(df: pd.DataFrame, n: int = 5) -> str:
    """Build a text representation of the first n rows."""
    return df.head(n).to_string()


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=_APP_TITLE,
        page_icon="🔍",
        layout="wide",
    )

    st.title(f"🔍 {_APP_TITLE}")
    st.caption("Upload a dataset to get an instant deterministic profile and AI-powered analysis.")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Supported formats: CSV, Excel (.xlsx/.xls), Parquet",
        )
        st.divider()
        st.header("⚙️ Settings")
        enable_ai = st.toggle(
            "Enable AI Analysis",
            value=True,
            help="Uses local Ollama or Gemini API (set LLM_PROVIDER in .env)",
        )

    if uploaded_file is None:
        _render_landing()
        return

    # ---- Load ----
    loader = DataLoader()
    try:
        df = loader.load_from_upload(uploaded_file.read(), uploaded_file.name)
    except UnsupportedFormatError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Failed to load file: {exc}")
        return

    # ---- Profile ----
    profiler = DataProfiler()
    with st.spinner("Computing profile…"):
        profile = profiler.profile(df)

    # Reset AI state when a different file is uploaded
    if st.session_state.get("last_file") != uploaded_file.name:
        st.session_state["last_file"] = uploaded_file.name
        st.session_state["ai_description"] = None
        st.session_state["analyst_insights"] = None
        st.session_state["analyst_markdown"] = None

    # ---- KPI row ----
    _render_kpi_row(profile)

    st.divider()

    # ---- Tabs ----
    tab_overview, tab_columns, tab_sample, tab_ai, tab_insights = st.tabs(
        ["📊 Overview", "📋 Columns", "🗂️ Sample Data", "🤖 AI Analysis", "💡 Insights"]
    )

    with tab_overview:
        _render_overview(profile)

    with tab_columns:
        _render_columns(profile)

    with tab_sample:
        st.dataframe(df.head(_SAMPLE_ROWS), use_container_width=True)

    with tab_ai:
        _render_ai_analysis(profile, enable_ai)

    with tab_insights:
        _render_insights(df, profile, enable_ai)


# ---------------------------------------------------------------------------
# Rendering helpers (existing)
# ---------------------------------------------------------------------------


def _render_landing() -> None:
    st.info("👆 Upload a CSV, Excel, or Parquet file using the sidebar to get started.")
    st.markdown(
        """
        ### What you'll get
        - **Instant profile** — shape, column types, missing values, duplicates, statistics
        - **Column-level analysis** — distributions, top values, numeric stats
        - **AI-powered description** — structured markdown report from an LLM
        - **Automated insights** — trend, anomaly, correlation, and distribution detection
        """
    )


def _render_kpi_row(profile: DataProfile) -> None:
    cols = st.columns(5)
    cols[0].metric("Rows", f"{profile.shape[0]:,}")
    cols[1].metric("Columns", f"{profile.shape[1]:,}")
    cols[2].metric("Duplicates", f"{profile.duplicates_count:,}")
    cols[3].metric("Missing", f"{profile.total_missing_pct:.1f} %")
    cols[4].metric("Memory", f"{profile.memory_mb:.2f} MB")


def _render_overview(profile: DataProfile) -> None:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Column Types")
        type_breakdown = {
            "Numeric": profile.numeric_cols,
            "Categorical": profile.categorical_cols,
            "Datetime": profile.datetime_cols,
            "Boolean": profile.boolean_cols,
            "Other": profile.other_cols,
        }
        for label, count in type_breakdown.items():
            if count > 0:
                st.write(f"- **{label}**: {count}")

    with col_right:
        st.subheader("Missing Values by Column")
        missing_items = [(col, pct) for col, pct in profile.missing.items() if pct > 0]
        if not missing_items:
            st.success("No missing values detected.")
        else:
            missing_df = pd.DataFrame(missing_items, columns=["Column", "Missing %"]).sort_values(
                "Missing %", ascending=False
            )
            st.dataframe(missing_df, use_container_width=True, hide_index=True)


def _render_columns(profile: DataProfile) -> None:
    for col_name, cp in profile.columns.items():
        header = f"**{col_name}** — `{cp.dtype}` ({cp.dtype_category})"
        with st.expander(header, expanded=False):
            meta_cols = st.columns(3)
            meta_cols[0].metric("Missing", f"{cp.missing_pct:.1f} %")
            meta_cols[1].metric("Unique values", f"{cp.unique_count:,}")
            meta_cols[2].metric("Unique %", f"{cp.unique_pct:.1f} %")

            if cp.dtype_category == "numeric":
                num_cols = st.columns(4)
                num_cols[0].metric("Min", cp.min)
                num_cols[1].metric("Max", cp.max)
                num_cols[2].metric(
                    "Mean",
                    f"{cp.mean:.4f}" if cp.mean is not None else "N/A",
                )
                num_cols[3].metric(
                    "Std",
                    f"{cp.std:.4f}" if cp.std is not None else "N/A",
                )
                st.caption(
                    f"Median: {cp.median} | Q25: {cp.q25} | Q75: {cp.q75} | "
                    f"Skewness: {cp.skewness} | Kurtosis: {cp.kurtosis}"
                )
                if cp.zeros_count:
                    st.caption(f"Zeros: {cp.zeros_count} ({cp.zeros_pct:.1f} %)")

            elif cp.dtype_category in ("categorical", "boolean") and cp.top_values:
                st.caption("Top values:")
                tv_df = pd.DataFrame(cp.top_values.items(), columns=["Value", "Count"])
                st.dataframe(tv_df, use_container_width=True, hide_index=True)

            elif cp.dtype_category == "datetime":
                st.caption(f"Range: {cp.min_date} → {cp.max_date} ({cp.date_range_days} days)")


def _render_ai_analysis(profile: DataProfile, enable_ai: bool) -> None:
    if not enable_ai:
        st.info("Enable **AI Analysis** in the sidebar to use this feature.")
        return

    if st.button("🚀 Generate AI Analysis", type="primary"):
        agent = ProfilerAgent()
        with st.spinner("Generating AI analysis… this may take a moment."):
            try:
                description = agent.describe(profile)
                st.session_state["ai_description"] = description
            except Exception as exc:
                st.error(f"AI analysis failed: {exc}")
                st.info(
                    "Make sure Ollama is running locally, or set `LLM_PROVIDER=gemini` "
                    "with a valid `GOOGLE_API_KEY` in your `.env` file."
                )
                return

    if st.session_state.get("ai_description"):
        st.markdown(st.session_state["ai_description"])
    else:
        st.caption("Click **Generate AI Analysis** to produce an LLM-powered report.")


# ---------------------------------------------------------------------------
# Phase 2: Insights tab
# ---------------------------------------------------------------------------

# Category colors for the badge-like display
_CATEGORY_COLORS = {
    "trend": "#2196F3",
    "anomaly": "#FF5722",
    "correlation": "#9C27B0",
    "distribution": "#4CAF50",
    "general": "#607D8B",
}

_CATEGORY_ICONS = {
    "trend": "📈",
    "anomaly": "⚠️",
    "correlation": "🔗",
    "distribution": "📊",
    "general": "💡",
}

_PRIORITY_COLORS = {
    "high": "🔴",
    "medium": "🟡",
    "low": "🟢",
}


def _render_insights(df: pd.DataFrame, profile: DataProfile, enable_ai: bool) -> None:
    """Render the Phase 2 Insights tab."""
    if not enable_ai:
        st.info("Enable **AI Analysis** in the sidebar to use this feature.")
        return

    # Check if profiler AI analysis exists (used as input)
    profiler_output = st.session_state.get("ai_description")

    if not profiler_output:
        st.warning(
            "Please generate the **AI Analysis** first (previous tab). "
            "The Analyst agent needs the profiler output as input."
        )
        return

    if st.button("🔍 Generate Insights", type="primary", key="btn_insights"):
        profile_data = _profile_to_dict(profile)
        sample_text = _build_sample_text(df)

        agent = AnalystAgent(temperature=0.5)

        with st.spinner("Generating insights… this may take a moment."):
            try:
                result = agent.run(
                    profiler_output=profiler_output,
                    sample_text=sample_text,
                    profile_data=profile_data,
                )

                if result.get("error"):
                    st.error(f"Insight generation failed: {result['error']}")
                    return

                st.session_state["analyst_insights"] = result.get("insights", [])
                st.session_state["analyst_markdown"] = result.get("analyst_output", "")

            except Exception as exc:
                st.error(f"Insight generation failed: {exc}")
                st.info("Make sure Ollama is running locally (`ollama serve`).")
                return

    # ---- Display insights ----
    insights = st.session_state.get("analyst_insights")
    markdown = st.session_state.get("analyst_markdown")

    if not insights:
        st.caption("Click **Generate Insights** to produce AI-powered data insights.")
        return

    # Summary metrics
    _render_insight_summary(insights)

    st.divider()

    # View toggle
    view_mode = st.radio(
        "Display mode",
        ["Cards", "Markdown"],
        horizontal=True,
        key="insight_view_mode",
    )

    if view_mode == "Cards":
        _render_insight_cards(insights)
    else:
        st.markdown(markdown)


def _render_insight_summary(insights: list[dict]) -> None:
    """Render summary KPI row for insights."""
    total = len(insights)
    high = sum(1 for i in insights if i.get("priority") == "high")
    medium = sum(1 for i in insights if i.get("priority") == "medium")
    low = sum(1 for i in insights if i.get("priority") == "low")

    # Count categories
    categories = {}
    for ins in insights:
        cat = ins.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1

    cols = st.columns(4)
    cols[0].metric("Total Insights", total)
    cols[1].metric("🔴 High Priority", high)
    cols[2].metric("🟡 Medium Priority", medium)
    cols[3].metric("🟢 Low Priority", low)

    # Category breakdown
    if categories:
        cat_text = " · ".join(
            f"{_CATEGORY_ICONS.get(cat, '💡')} {cat.capitalize()}: {count}"
            for cat, count in categories.items()
        )
        st.caption(f"**Categories**: {cat_text}")


def _render_insight_cards(insights: list[dict]) -> None:
    """Render each insight as an expandable card."""
    # Group by category
    by_category: dict[str, list[dict]] = {}
    for insight in insights:
        cat = insight.get("category", "general")
        by_category.setdefault(cat, []).append(insight)

    for category, cat_insights in by_category.items():
        icon = _CATEGORY_ICONS.get(category, "💡")
        st.subheader(f"{icon} {category.capitalize()}")

        for insight in cat_insights:
            priority_icon = _PRIORITY_COLORS.get(insight.get("priority", "medium"), "⚪")
            header = f"{priority_icon} {insight['title']}"

            with st.expander(header, expanded=True):
                st.markdown(f"**Observation**: {insight.get('observation', 'N/A')}")
                st.markdown(f"**Hypothesis**: {insight.get('hypothesis', 'N/A')}")
                st.markdown(f"**Recommendation**: {insight.get('recommendation', 'N/A')}")

                tag_cols = st.columns(2)
                tag_cols[0].caption(
                    f"Priority: **{insight.get('priority', 'medium').capitalize()}**"
                )
                tag_cols[1].caption(
                    f"Category: **{insight.get('category', 'general').capitalize()}**"
                )


if __name__ == "__main__":
    main()
