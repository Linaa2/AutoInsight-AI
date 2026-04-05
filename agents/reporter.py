"""
agents/reporter.py — Phase: Report generation.

Architecture:
    - ReporterAgent (class): Generates a full analysis report from profiler output,
      analyst insights, and visualizer chart titles.
    - reporter_node(): LangGraph node entry point (delegates to ReporterAgent).

Input (from LangGraph state):
    - profiler_output    : str          (profiler markdown)
    - analyst_output     : str          (analyst markdown)
    - insights           : list[dict]   (structured insight objects)
    - visualizer_output  : dict | None  (charts metadata from visualizer)
    - critic_output      : str          (critic markdown, from CriticAgent)
    - uncertainty_output : str          (confidence scores, from UncertaintyEstimator)

Output:
    - reporter_output    : str          (full markdown report)
"""

import logging

from agents.context_digest import (
    build_confidence_digest,
    build_critiques_digest,
    build_profile_digest,
)
from utils.llm import call_llm_with_messages
from utils.prompt_loader import load_prompt_section

logger = logging.getLogger(__name__)


# Load prompts through canonical loader
PROMPTS: dict[str, dict[str, str]] = {"reporter": load_prompt_section("reporter")}


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def extract_chart_titles(visualizer_output: dict | None) -> list[str]:
    """Extract chart titles from the visualizer output dict."""
    if not visualizer_output or not isinstance(visualizer_output, dict):
        return []
    charts = visualizer_output.get("charts", [])
    titles: list[str] = []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        raw_spec = chart.get("spec")
        spec: dict = raw_spec if isinstance(raw_spec, dict) else chart
        titles.append(spec.get("title", "Untitled chart"))
    return titles


def build_charts_summary(visualizer_output: dict | None) -> str:
    """Build a readable summary of generated charts for the LLM context."""
    titles = extract_chart_titles(visualizer_output)
    if not titles:
        return "No charts were generated."
    lines = [f"- Chart: {title}" for title in titles]
    return "\n".join(lines)


def build_insights_summary(insights: list[dict] | None) -> str:
    """Build a compact text summary of structured insights for the LLM context."""
    if not insights:
        return "No structured insights available."

    lines = []
    for i, ins in enumerate(insights, 1):
        priority = ins.get("priority", "medium").capitalize()
        category = ins.get("category", "general").capitalize()
        lines.append(
            f"{i}. [{priority}] [{category}] {ins.get('title', 'Untitled')}\n"
            f"   Observation: {ins.get('observation', 'N/A')}\n"
            f"   Recommendation: {ins.get('recommendation', 'N/A')}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  REPORTER AGENT (core class)
# ═══════════════════════════════════════════════════════════════════════════


class ReporterAgent:
    """
    Agent that generates a full analysis report.

    Synthesizes outputs from all previous agents (profiler, analyst, visualizer)
    into a coherent, professional markdown report.

    Workflow:
        1. Collect profiler output, analyst insights, and chart info.
        2. Build context for the LLM.
        3. Call LLM (text model) to generate the report.
        4. Return the markdown report.

    Usage:
        agent = ReporterAgent()
        result = agent.run(
            profiler_output="...",
            analyst_output="...",
            insights=[...],
            visualizer_output={...},
        )
        print(result["reporter_output"])
    """

    def run(
        self,
        profiler_output: str,
        analyst_output: str,
        insights: list[dict] | None = None,
        visualizer_output: dict | None = None,
        rag_context: str = "",
        critic_output: str = "",
        uncertainty_output: str = "",
        profile_data: dict | None = None,
        critiques: list[dict] | None = None,
        confidence_scores: list[dict] | None = None,
        callbacks: list | None = None,
    ) -> dict:
        """
        Generate the full analysis report.

        Args:
            profiler_output:  Profiler markdown output.
            analyst_output:   Analyst markdown output.
            insights:         Structured insight dicts (optional, enriches context).
            visualizer_output: Visualizer output dict with chart metadata (optional).
            rag_context:      Optional context retrieved from ChromaDB (previous runs).
            critic_output:    Critic markdown (adversarial review of insights).
            uncertainty_output: Confidence scores markdown from Uncertainty Estimator.
            callbacks:        Optional LangChain callbacks (e.g. LangFuse handler).

        Returns:
            Dict with 'reporter_output' (str): the full markdown report.
        """
        try:
            charts_summary = build_charts_summary(visualizer_output)
            insights_summary = build_insights_summary(insights)
            profile_summary = (
                build_profile_digest(profile_data) if profile_data else profiler_output
            )
            analyst_context = (
                f"{len(insights or [])} structured insights are summarized below."
                if insights
                else analyst_output
            )
            critique_summary = (
                build_critiques_digest(critiques)
                if critiques
                else critic_output or "No critique summary available."
            )
            confidence_summary = (
                build_confidence_digest(confidence_scores)
                if confidence_scores
                else uncertainty_output or "No confidence summary available."
            )

            system_prompt = PROMPTS["reporter"]["system"]
            human_prompt = PROMPTS["reporter"]["human"].format(
                profiler_output=profile_summary or profiler_output,
                analyst_output=analyst_context,
                insights_summary=insights_summary,
                charts_summary=charts_summary,
            )

            # Append prior-run context if available (lightweight, controlled injection)
            if rag_context and rag_context.strip():
                human_prompt += (
                    "\n\n## 📚 Context from Previous Analyses\n"
                    "The following is relevant context retrieved from a previous run "
                    "on this dataset. Use it only if it adds value to the current report.\n\n"
                    + rag_context.strip()
                )

            # Append critic review when available
            if critique_summary and critique_summary.strip():
                human_prompt += (
                    "\n\n## 🔎 Critic Review\n"
                    "Each insight was reviewed by an adversarial critic. Integrate "
                    "the critique into the Key Insights section — mention weaknesses "
                    "or alternative explanations where the verdict is "
                    "'partially_supported' or 'weak'.\n\n" + critique_summary.strip()
                )

            # Append uncertainty confidence scores when available
            if confidence_summary and confidence_summary.strip():
                human_prompt += (
                    "\n\n## 🎯 Insight Confidence Scores\n"
                    "The following table shows the data-driven confidence level for each "
                    "insight. When writing the Key Insights section, qualify insights with "
                    "low or medium confidence accordingly.\n\n" + confidence_summary.strip()
                )

            raw = call_llm_with_messages(
                system=system_prompt,
                human=human_prompt,
                callbacks=callbacks,
                task="reporter",
            )

            logger.info(f"ReporterAgent: report generated ({len(raw)} chars)")

            return {"reporter_output": raw}

        except Exception as e:
            logger.error(f"Error in ReporterAgent.run: {e}", exc_info=True)
            return {
                "reporter_output": f"❌ Error during report generation: {e}",
                "error": str(e),
            }


# ═══════════════════════════════════════════════════════════════════════════
#  LANGGRAPH NODE (pipeline entry point)
# ═══════════════════════════════════════════════════════════════════════════


def reporter_node(state: dict) -> dict:
    """
    LangGraph node — Generate the final analysis report.

    Reads from state:
        - profiler_output (str)
        - analyst_output (str)
        - insights (list[dict], optional)
        - visualizer_output (dict, optional)
        - critic_output (str, optional)
        - uncertainty_output (str, optional)

    Writes to state:
        - reporter_output (str): full markdown report
    """
    profiler_output = state.get("profiler_output", "")
    analyst_output = state.get("analyst_output", "")
    insights = state.get("insights")
    visualizer_output = state.get("visualizer_output")
    critic_output = state.get("critic_output", "")
    uncertainty_output = state.get("uncertainty_output", "")

    if not profiler_output and not analyst_output:
        return {
            "reporter_output": "⚠️ No input data for report generation.",
            "error": "No profiler_output or analyst_output provided",
        }

    agent = ReporterAgent()
    return agent.run(
        profiler_output=profiler_output,
        analyst_output=analyst_output,
        insights=insights,
        visualizer_output=visualizer_output,
        critic_output=critic_output,
        uncertainty_output=uncertainty_output,
        profile_data=state.get("profile_data"),
        critiques=state.get("critiques"),
        confidence_scores=state.get("confidence_scores"),
    )
