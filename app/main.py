"""AutoInsight AI — Streamlit application entry point (P1: Profiler UI)."""

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that local modules are importable
# when the app is launched with `streamlit run app/main.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.profiler import ProfilerAgent
from tools.data_loader import DataLoader, UnsupportedFormatError
from tools.profiler_engine import DataProfile, DataProfiler
from utils.logger import get_module_logger

load_dotenv()

logger = get_module_logger(__name__, console=False)

_APP_TITLE = os.getenv("APP_TITLE", "AutoInsight AI")
_SAMPLE_ROWS = int(os.getenv("PROFILER_SAMPLE_ROWS", "5"))


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
    logger.info("File uploaded: '%s'", uploaded_file.name)
    loader = DataLoader()
    try:
        df = loader.load_from_upload(uploaded_file.read(), uploaded_file.name)
    except UnsupportedFormatError as exc:
        logger.warning("Unsupported format for '%s': %s", uploaded_file.name, exc)
        st.error(str(exc))
        return
    except Exception as exc:
        logger.error("Failed to load '%s': %s", uploaded_file.name, exc)
        st.error(f"Failed to load file: {exc}")
        return
    logger.info("File loaded: %d rows x %d cols", df.shape[0], df.shape[1])

    # ---- Profile ----
    profiler = DataProfiler()
    with st.spinner("Computing profile…"):
        profile = profiler.profile(df)
    logger.info("Profile computed for '%s'", uploaded_file.name)

    # Reset AI description when a different file is uploaded
    if st.session_state.get("last_file") != uploaded_file.name:
        st.session_state["last_file"] = uploaded_file.name
        st.session_state["ai_description"] = None

    # ---- KPI row ----
    _render_kpi_row(profile)

    st.divider()

    # ---- Tabs ----
    tab_overview, tab_columns, tab_sample, tab_ai = st.tabs(
        ["📊 Overview", "📋 Columns", "🗂️ Sample Data", "🤖 AI Analysis"]
    )

    with tab_overview:
        _render_overview(profile)

    with tab_columns:
        _render_columns(profile)

    with tab_sample:
        st.dataframe(df.head(_SAMPLE_ROWS), use_container_width=True)

    with tab_ai:
        _render_ai_analysis(profile, enable_ai)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_landing() -> None:
    st.info("👆 Upload a CSV, Excel, or Parquet file using the sidebar to get started.")
    st.markdown(
        """
        ### What you'll get
        - **Instant profile** — shape, column types, missing values, duplicates, statistics
        - **Column-level analysis** — distributions, top values, numeric stats
        - **AI-powered description** — structured markdown report from an LLM
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
        logger.info("AI analysis requested")
        agent = ProfilerAgent()
        with st.spinner("Generating AI analysis… this may take a moment."):
            try:
                description = agent.describe(profile)
                st.session_state["ai_description"] = description
                logger.info("AI analysis complete (%d chars)", len(description))
            except Exception as exc:
                logger.error("AI analysis failed: %s", exc)
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


if __name__ == "__main__":
    main()
