"""AutoInsight AI - Streamlit application entry point (P1-P3: Profiler/Analyst/Reporter + P7: Evaluation UI)."""

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.analyst import AnalystAgent
from agents.profiler import ProfilerAgent
from agents.reporter import ReporterAgent
from evaluation.llm_judge import EvaluationAgent
from evaluation.schemas import AnalystEvaluationResult, EvaluationResult, PipelineEvaluation
from tools.data_loader import DataLoader, UnsupportedFormatError
from tools.profiler_engine import DataProfile, DataProfiler

load_dotenv()

_APP_TITLE = os.getenv("APP_TITLE", "AutoInsight AI")
_SAMPLE_ROWS = int(os.getenv("PROFILER_SAMPLE_ROWS", "5"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile_to_dict(profile: DataProfile) -> dict:
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
    return str(df.head(n).to_string())


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
            help="Uses local Ollama or Gemini API",
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

    # Reset state when new file
    if st.session_state.get("last_file") != uploaded_file.name:
        st.session_state["last_file"] = uploaded_file.name
        st.session_state["ai_description"] = None
        st.session_state["analyst_insights"] = None
        st.session_state["analyst_markdown"] = None
        st.session_state["reporter_output"] = None
        st.session_state["pipeline_eval"] = None  # auto-invalidate on new file

    # ---- KPI row ----
    _render_kpi_row(profile)

    st.divider()

    # ---- Tabs ----
    tab_overview, tab_columns, tab_sample, tab_ai, tab_insights, tab_report, tab_eval = st.tabs(
        [
            "📊 Overview",
            "📋 Columns",
            "🗂️ Sample Data",
            "🤖 AI Analysis",
            "💡 Insights",
            "📄 Report",
            "🔍 Evaluation",
        ]
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

    with tab_report:
        _render_report(enable_ai)

    with tab_eval:
        _render_evaluation(profile, enable_ai)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_landing() -> None:
    st.info("👆 Upload a CSV, Excel, or Parquet file using the sidebar to get started.")


def _render_kpi_row(profile: DataProfile) -> None:
    cols = st.columns(5)
    cols[0].metric("Rows", f"{profile.shape[0]:,}")
    cols[1].metric("Columns", f"{profile.shape[1]:,}")
    cols[2].metric("Duplicates", f"{profile.duplicates_count:,}")
    cols[3].metric("Missing", f"{profile.total_missing_pct:.1f} %")
    cols[4].metric("Memory", f"{profile.memory_mb:.2f} MB")


def _render_overview(_profile: DataProfile) -> None:
    st.subheader("Column Types")


def _render_columns(_profile: DataProfile) -> None:
    st.subheader("Columns")


# ---------------------------------------------------------------------------
# AI Analysis (UNCHANGED)
# ---------------------------------------------------------------------------


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
                return

    if st.session_state.get("ai_description"):
        st.markdown(st.session_state["ai_description"])
    else:
        st.caption("Click **Generate AI Analysis** to produce an LLM-powered report.")


# ---------------------------------------------------------------------------
# Insights (UNCHANGED)
# ---------------------------------------------------------------------------

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
    if not enable_ai:
        st.info("Enable **AI Analysis**")
        return

    profiler_output = st.session_state.get("ai_description")

    if not profiler_output:
        st.warning("Generate AI Analysis first")
        return

    if st.button("🔍 Generate Insights", type="primary", key="btn_insights"):
        agent = AnalystAgent()

        with st.spinner("Generating insights… this may take a moment."):
            result = agent.run(
                profiler_output=profiler_output,
                sample_text=_build_sample_text(df),
                profile_data=_profile_to_dict(profile),
            )

        st.session_state["analyst_insights"] = result.get("insights", [])
        st.session_state["analyst_markdown"] = result.get("analyst_output", "")

    insights = st.session_state.get("analyst_insights")
    markdown = st.session_state.get("analyst_markdown")

    if not insights:
        st.caption("Generate insights")
        return

    view_mode = st.radio("Display mode", ["Cards", "Markdown"], horizontal=True)

    if view_mode == "Markdown":
        st.markdown(markdown)
    else:
        for ins in insights:
            with st.expander(ins["title"], expanded=True):
                st.write("**Observation:**", ins.get("observation"))
                st.write("**Hypothesis:**", ins.get("hypothesis"))
                st.write("**Recommendation:**", ins.get("recommendation"))


# ---------------------------------------------------------------------------
# Reporter (NEW)
# ---------------------------------------------------------------------------


def _render_report(enable_ai: bool) -> None:
    if not enable_ai:
        st.info("Enable **AI Analysis**")
        return

    profiler_output = st.session_state.get("ai_description")
    analyst_output = st.session_state.get("analyst_markdown")
    insights = st.session_state.get("analyst_insights")

    if not profiler_output:
        st.warning("Generate AI Analysis first")
        return

    if not analyst_output:
        st.warning("Generate Insights first")
        return

    if st.button("📄 Generate Full Report", type="primary", key="btn_report"):
        agent = ReporterAgent()

        with st.spinner("Generating report… this may take a moment."):
            result = agent.run(
                profiler_output=profiler_output,
                analyst_output=analyst_output,
                insights=insights,
            )

        if result.get("error"):
            st.error(result["error"])
            return

        st.session_state["reporter_output"] = result.get("reporter_output", "")

    report = st.session_state.get("reporter_output")

    if not report:
        st.caption("Click **Generate Full Report**")
        return

    st.markdown(report)

    st.divider()

    st.download_button(
        label="⬇️ Download Report (.md)",
        data=report,
        file_name="analysis_report.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Evaluation (P7)
# ---------------------------------------------------------------------------

_GRADE_COLORS = {
    "excellent": "🟢",
    "good": "🔵",
    "fair": "🟠",
    "poor": "🔴",
}


def _score_bar(label: str, score: float) -> None:
    """Render a labelled progress bar for a single criterion score."""
    grade_icon = _GRADE_COLORS.get(EvaluationResult.compute_grade(score), "⚪")
    st.write(f"**{label}** {grade_icon} `{score:.2f}`")
    st.progress(score)


def _render_artifact_panel(title: str, result: EvaluationResult) -> None:
    """Render one agent evaluation panel inside a Streamlit column."""
    grade_icon = _GRADE_COLORS.get(result.grade, "⚪")
    st.metric(
        label=title,
        value=f"{result.overall_score:.2f}",
        delta=result.grade,
    )
    st.caption(f"{grade_icon} {result.grade.capitalize()}")

    for criterion in result.criteria:
        _score_bar(criterion.label, criterion.score)
        if criterion.rationale:
            st.caption(criterion.rationale)

    if result.critique:
        st.write("**Critique:**", result.critique)

    if result.suggestions:
        st.write("**Suggestions:**")
        for s in result.suggestions:
            st.write(f"- {s}")

    # Per-insight table for analyst results
    if isinstance(result, AnalystEvaluationResult) and result.per_insight:
        with st.expander("Per-insight scores", expanded=False):
            rows = [
                {
                    "Insight": ins.title,
                    "Factual": round(ins.factual_correctness, 2),
                    "Relevance": round(ins.relevance, 2),
                    "Actionability": round(ins.actionability, 2),
                    "Note": ins.note,
                }
                for ins in result.per_insight
            ]
            st.dataframe(rows, use_container_width=True)


def _render_evaluation(profile: DataProfile, enable_ai: bool) -> None:
    if not enable_ai:
        st.info("Enable **AI Analysis** in the sidebar to use this feature.")
        return

    profiler_output = st.session_state.get("ai_description")
    analyst_insights = st.session_state.get("analyst_insights")
    analyst_markdown = st.session_state.get("analyst_markdown")
    reporter_output = st.session_state.get("reporter_output")

    has_any = any([profiler_output, analyst_insights, reporter_output])

    if not has_any:
        st.info(
            "Run at least one pipeline step (AI Analysis, Insights, or Report) before evaluating."
        )
        return

    if st.button("🔍 Run Evaluation", type="primary", key="btn_eval"):
        agent = EvaluationAgent()
        _new_eval = PipelineEvaluation()

        with st.spinner("Evaluating pipeline outputs… this may take a moment."):
            if profiler_output:
                try:
                    _new_eval.profiler_eval = agent.evaluate_profiler(profiler_output, profile)
                except Exception as exc:
                    st.error(f"Profiler evaluation failed: {exc}")

            if analyst_insights:
                try:
                    _new_eval.analyst_eval = agent.evaluate_analyst(analyst_insights, profile)
                except Exception as exc:
                    st.error(f"Analyst evaluation failed: {exc}")

            if reporter_output and analyst_markdown:
                try:
                    _new_eval.reporter_eval = agent.evaluate_reporter(
                        reporter_output, analyst_markdown
                    )
                except Exception as exc:
                    st.error(f"Reporter evaluation failed: {exc}")

        st.session_state["pipeline_eval"] = _new_eval

    pipeline_eval: PipelineEvaluation | None = st.session_state.get("pipeline_eval")

    if pipeline_eval is None:
        st.caption("Click **Run Evaluation** to score all available pipeline outputs.")
        return

    # ---- Pipeline-level score ----
    st.divider()
    pipeline_score = pipeline_eval.pipeline_score
    grade_icon = _GRADE_COLORS.get(pipeline_eval.pipeline_grade, "⚪")
    st.subheader(f"Pipeline Score: {pipeline_score:.2f} {grade_icon}")
    st.progress(pipeline_score)
    st.divider()

    # ---- Per-agent panels ----
    evals = [
        ("Profiler Quality", pipeline_eval.profiler_eval),
        ("Analyst Quality", pipeline_eval.analyst_eval),
        ("Reporter Quality", pipeline_eval.reporter_eval),
    ]
    available = [(title, res) for title, res in evals if res is not None]

    if not available:
        st.warning("No evaluation results available.")
        return

    cols = st.columns(len(available))
    for col, (title, result) in zip(cols, available):
        with col:
            _render_artifact_panel(title, result)


if __name__ == "__main__":
    main()
