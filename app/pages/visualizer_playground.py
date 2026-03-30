"""Visualizer Playground — Streamlit testing page for Phase 3.

This page allows developers to test the visualizer agent against predefined
dataset examples without running the Profiler or Analyst agents live.

User flow:
    1. Select a predefined example from the sidebar selectbox.
    2. Inspect the DataFrame preview, profiler output, and analyst insights.
    3. Click "Generate Visualizations" to invoke the visualizer pipeline.
    4. Inspect raw LLM output, parsed specs, generated code, and rendered figures.

Run directly (bypassing the main app):
    uv run streamlit run app/pages/visualizer_playground.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from agents.visualizer import generate_visualizations
from app.examples_visualizer import VisualizerExample, get_examples

if TYPE_CHECKING:
    from visualization.schemas import VisualizationPipelineResult

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Visualizer Playground — AutoInsight-AI",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helper: example selector
# ---------------------------------------------------------------------------


def _select_example(examples: list[VisualizerExample]) -> VisualizerExample:
    """Render a selectbox and return the chosen example dict."""
    names = [ex["name"] for ex in examples]
    choice = st.selectbox("Select a dataset example", options=names)
    return next(ex for ex in examples if ex["name"] == choice)


# ---------------------------------------------------------------------------
# Helper: render context (df + profiler + analyst)
# ---------------------------------------------------------------------------


def render_example_context(example: VisualizerExample) -> None:
    """Display the dataset preview, profiler output, and analyst insights."""
    # ---- Dataset preview ----
    st.subheader("Dataset Preview")
    st.dataframe(example["df"], use_container_width=True)

    col_left, col_right = st.columns(2)

    # ---- Profiler output ----
    with col_left:
        st.subheader("Profiler Output")
        st.markdown(
            f"```\n{example['profiler_output'].strip()}\n```",
            unsafe_allow_html=False,
        )

    # ---- Analyst insights ----
    with col_right:
        st.subheader("Analyst Insights")
        st.markdown(example["analyst_output"].strip())


# ---------------------------------------------------------------------------
# Helper: run the pipeline
# ---------------------------------------------------------------------------


def run_visualizer_example(example: VisualizerExample) -> VisualizationPipelineResult:
    """Call the visualizer pipeline for the given example and return results."""
    return generate_visualizations(
        df=example["df"],
        profile_summary=example["profiler_output"],
        insights_text=example["analyst_output"],
        columns_info=example.get("columns_info", ""),
    )


# ---------------------------------------------------------------------------
# Helper: render results
# ---------------------------------------------------------------------------


def render_visualizer_results(output: VisualizationPipelineResult) -> None:
    """Render the full visualizer output: raw LLM text, specs, figures, errors."""

    # ---- Top-level parsing status ----
    if output.parsing_error and not output.charts:
        # Complete failure — nothing to show
        st.error(f"Pipeline error: {output.parsing_error}")
    elif output.parsing_error:
        # Partial failure — some charts were produced
        st.warning(f"Partial parsing error: {output.parsing_error}")
    else:
        st.success(f"{len(output.charts)} chart(s) generated successfully.")

    # ---- Raw LLM output (always shown for debugging) ----
    with st.expander("Raw LLM output", expanded=False):
        raw = output.raw_llm_output or "(empty — LLM call may have failed)"
        st.code(raw, language="json")

    if not output.charts:
        st.info("No charts to display.")
        return

    # ---- Parsed specs overview ----
    with st.expander("Parsed chart specs (JSON)", expanded=False):
        import dataclasses
        import json

        specs_data = [dataclasses.asdict(cr.spec) for cr in output.charts]
        st.code(json.dumps(specs_data, indent=2), language="json")

    # ---- Per-chart rendering ----
    st.subheader("Generated Charts")

    for i, chart_result in enumerate(output.charts, start=1):
        spec = chart_result.spec
        execution = chart_result.execution

        title = spec.title
        chart_type = spec.chart_type
        explanation = spec.explanation or ""
        code = spec.code

        st.markdown(f"#### {i}. {title}  `{chart_type}`")

        if explanation:
            st.markdown(f"_{explanation}_")

        # Status badge
        if execution.success:
            st.success("Execution succeeded")
        else:
            st.error(f"Execution failed: {execution.error}")

        # Generated code (always visible for debugging)
        with st.expander("Generated code", expanded=not execution.success):
            st.code(code, language="python")

        # Plotly figure
        if execution.success and execution.figure is not None:
            st.plotly_chart(execution.figure, use_container_width=True)
        elif not execution.success:
            st.warning("Figure could not be rendered due to execution failure.")

        st.divider()


# ---------------------------------------------------------------------------
# Main page layout
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Visualizer Playground page."""
    st.title("📊 Visualizer Playground")
    st.markdown(
        """
        **Phase 3 — Development Testing Page**

        This page lets you test the visualizer agent against predefined dataset examples.
        Select an example, review the context, then click the button to generate charts.
        All intermediate results (raw LLM output, parsed JSON, generated code) are
        available in the expandable sections for debugging.
        """
    )

    st.divider()

    # --- Example selection ---
    examples = get_examples()
    selected = _select_example(examples)

    st.divider()

    # --- Context preview ---
    render_example_context(selected)

    st.divider()

    # --- Run button ---
    if st.button("🚀 Generate Visualizations", type="primary"):
        with st.spinner("Calling the visualizer agent — this may take a few seconds..."):
            output = run_visualizer_example(selected)

        st.divider()
        render_visualizer_results(output)


main()
