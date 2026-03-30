"""
agents/analyst.py — Phase 2: Insight generation and structuring.

Architecture:
    - AnalystAgent (class): Core agent that generates and structures insights.
    - InsightCategorizer (class): Categorizes insights (keywords or LLM).
    - InsightFormatter (class): Formats insights to markdown.
    - Helper functions: JSON parsing, validation (pure functions).
    - analyst_node(): LangGraph node entry point (delegates to AnalystAgent).

Input (from LangGraph state):
    - profiler_output : str
    - profile_data    : dict
    - sample_text     : str

Output:
    - analyst_output  : str          (structured markdown)
    - insights        : list[dict]   (structured insight objects)
"""

import json
import logging
import re
from pathlib import Path
from typing import ClassVar

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm import get_langfuse_handler, get_llm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPT LOADER
# ═══════════════════════════════════════════════════════════════════════════


def _load_prompts() -> dict:
    """Load prompts from prompts.yaml at project root."""
    candidates = [
        Path(__file__).resolve().parent.parent / "prompts.yaml",
        Path.cwd() / "prompts.yaml",
    ]
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"prompts.yaml not found. Searched: {[str(p) for p in candidates]}")


PROMPTS = _load_prompts()


# ═══════════════════════════════════════════════════════════════════════════
#  PARSING HELPERS (pure functions)
# ═══════════════════════════════════════════════════════════════════════════


def extract_json(raw: str) -> dict | None:
    """Attempt to extract a JSON object from a potentially noisy LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.replace("```", "").strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    brace_count = 0
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    break
    return None


def validate_insight(insight: dict) -> bool:
    """Check that an insight has all required non-empty fields."""
    required = {"title", "observation", "hypothesis", "recommendation", "priority"}
    return all(key in insight and insight[key] for key in required)


def normalize_priority(priority: str) -> str:
    """Normalize priority value to high/medium/low."""
    p = priority.strip().lower()
    if p in ("high", "haute", "élevée", "critical"):
        return "high"
    if p in ("low", "basse", "faible"):
        return "low"
    return "medium"


def fallback_parse_markdown(raw: str) -> list[dict]:
    """
    Fallback parser: extract insights from structured markdown
    when the LLM does not respect JSON format.
    """
    insights = []
    current = {}

    for line in raw.split("\n"):
        line = line.strip()
        lower = line.lower()

        if lower.startswith("**title") or lower.startswith("- **title"):
            if current and "title" in current:
                insights.append(current)
            current = {}
            current["title"] = line.split(":", 1)[-1].strip().strip("*").strip()
        elif lower.startswith("**observation") or lower.startswith("- **observation"):
            current["observation"] = line.split(":", 1)[-1].strip().strip("*").strip()
        elif any(
            lower.startswith(p)
            for p in ("**hypothesis", "- **hypothesis", "**hypothèse", "- **hypothèse")
        ):
            current["hypothesis"] = line.split(":", 1)[-1].strip().strip("*").strip()
        elif lower.startswith("**recommendation") or lower.startswith("- **recommendation"):
            current["recommendation"] = line.split(":", 1)[-1].strip().strip("*").strip()
        elif any(
            lower.startswith(p)
            for p in ("**priority", "- **priority", "**priorité", "- **priorité")
        ):
            current["priority"] = line.split(":", 1)[-1].strip().strip("*").strip()

    if current and "title" in current:
        insights.append(current)

    for insight in insights:
        insight.setdefault("observation", "")
        insight.setdefault("hypothesis", "")
        insight.setdefault("recommendation", "")
        insight.setdefault("priority", "medium")

    return [i for i in insights if i.get("title")]


# ═══════════════════════════════════════════════════════════════════════════
#  INSIGHT CATEGORIZER (class)
# ═══════════════════════════════════════════════════════════════════════════


class InsightCategorizer:
    """Categorizes insights by keyword matching or LLM classification."""

    CATEGORY_KEYWORDS: ClassVar[dict[str, list[str]]] = {
        "trend": [
            "increase",
            "decrease",
            "growing",
            "declining",
            "evolution",
            "trend",
            "rise",
            "drop",
            "temporal",
            "time",
            "month",
            "year",
            "quarter",
            "progression",
            "seasonal",
            "augment",
            "diminue",
            "hausse",
            "baisse",
            "croissant",
        ],
        "anomaly": [
            "outlier",
            "anomal",
            "unexpected",
            "extreme",
            "abnormal",
            "spike",
            "crash",
            "exception",
            "gap",
            "unusual",
            "aberrant",
            "inattendu",
            "pic",
            "chute",
        ],
        "correlation": [
            "correlat",
            "link",
            "relation",
            "associat",
            "depend",
            "proportional",
            "inversely",
            "co-occur",
            "interact",
            "corrél",
            "lien",
        ],
        "distribution": [
            "distribut",
            "concentrat",
            "dispers",
            "imbalanc",
            "majority",
            "dominant",
            "proportion",
            "frequent",
            "rare",
            "répartit",
            "déséquilibr",
            "skew",
        ],
    }

    VALID_CATEGORIES: ClassVar[set[str]] = {
        "trend",
        "anomaly",
        "correlation",
        "distribution",
        "general",
    }

    def __init__(self, use_llm: bool = False, model: str | None = None):
        """
        Args:
            use_llm: If True, use LLM for categorization (more accurate, slower).
            model: Optional Ollama model override.
        """
        self.use_llm = use_llm
        self.model = model

    def categorize(self, insight: dict) -> str:
        """Assign a category to a single insight."""
        if self.use_llm:
            return self._categorize_with_llm(insight)
        return self._categorize_by_keywords(insight)

    def categorize_all(self, insights: list[dict]) -> list[dict]:
        """Add the 'category' key to each insight in the list."""
        for insight in insights:
            insight["category"] = self.categorize(insight)
        return insights

    def _categorize_by_keywords(self, insight: dict) -> str:
        """Quick keyword-based categorization."""
        text = (
            f"{insight.get('title', '')} "
            f"{insight.get('observation', '')} "
            f"{insight.get('hypothesis', '')}"
        ).lower()

        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            scores[category] = sum(1 for kw in keywords if kw in text)

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def _categorize_with_llm(self, insight: dict) -> str:
        """Categorize using LLM with fallback to keywords on failure."""
        try:
            llm = get_llm(temperature=0.0, model=self.model)
            handler = get_langfuse_handler()

            system_prompt = PROMPTS["categorizer"]["system"]
            human_prompt = PROMPTS["categorizer"]["human"].format(
                insight_json=json.dumps(insight, ensure_ascii=False, indent=2)
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]

            config = {}
            if handler:
                config["callbacks"] = [handler]

            response = llm.invoke(messages, config=config)
            cat = response.content.strip().lower().replace('"', "").replace("'", "")

            return cat if cat in self.VALID_CATEGORIES else self._categorize_by_keywords(insight)

        except Exception as e:
            logger.warning(f"LLM categorization failed ({e}), falling back to keywords")
            return self._categorize_by_keywords(insight)


# ═══════════════════════════════════════════════════════════════════════════
#  MARKDOWN FORMATTER (class)
# ═══════════════════════════════════════════════════════════════════════════


class InsightFormatter:
    """Formats structured insights into readable markdown."""

    CATEGORY_LABELS: ClassVar[dict[str, str]] = {
        "trend": "📈 Trend",
        "anomaly": "⚠️ Anomaly",
        "correlation": "🔗 Correlation",
        "distribution": "📊 Distribution",
        "general": "💡 General",
    }

    PRIORITY_ICONS: ClassVar[dict[str, str]] = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢",
    }

    @staticmethod
    def to_markdown(insights: list[dict]) -> str:
        """Convert a list of insights to structured markdown for the UI."""
        if not insights:
            return "⚠️ No insights could be generated."

        lines = ["# 🔍 Generated Insights\n"]

        # Group by category
        by_category: dict[str, list[dict]] = {}
        for insight in insights:
            cat = insight.get("category", "general")
            by_category.setdefault(cat, []).append(insight)

        for category, cat_insights in by_category.items():
            label = InsightFormatter.CATEGORY_LABELS.get(category, f"💡 {category.capitalize()}")
            lines.append(f"## {label}\n")

            for insight in cat_insights:
                icon = InsightFormatter.PRIORITY_ICONS.get(insight["priority"], "⚪")
                lines.append(f"### {icon} {insight['title']}\n")
                lines.append(f"**Observation**: {insight['observation']}\n")
                lines.append(f"**Hypothesis**: {insight['hypothesis']}\n")
                lines.append(f"**Recommendation**: {insight['recommendation']}\n")
                lines.append(f"**Priority**: {insight['priority'].capitalize()}\n")
                lines.append("---\n")

        # Summary
        total = len(insights)
        high_count = sum(1 for i in insights if i["priority"] == "high")
        lines.append(f"\n> **Summary**: {total} insights generated, {high_count} high priority.")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYST AGENT (core class)
# ═══════════════════════════════════════════════════════════════════════════


class AnalystAgent:
    """
    Agent that generates structured insights from a dataset profile.

    Workflow:
        1. Build context from profiler output, profile data, and sample.
        2. Call LLM to generate raw insights (JSON).
        3. Parse and validate the response.
        4. Categorize each insight.
        5. Format as markdown.

    Usage:
        agent = AnalystAgent(model="mistral", temperature=0.5)
        result = agent.run(profiler_output, sample_text, profile_data)
        # result = {"analyst_output": str, "insights": list[dict]}
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.5,
        use_llm_categorization: bool = False,
    ):
        """
        Args:
            model: Ollama model name (None = use env default).
            temperature: LLM generation temperature.
            use_llm_categorization: Use LLM for insight categorization.
        """
        self.model = model
        self.temperature = temperature
        self.categorizer = InsightCategorizer(use_llm=use_llm_categorization, model=model)
        self.formatter = InsightFormatter()

    def run(
        self,
        profiler_output: str,
        sample_text: str,
        profile_data: dict | None = None,
    ) -> dict:
        """
        Execute the full analyst pipeline.

        Args:
            profiler_output: Profiler markdown output.
            sample_text: Dataset sample as text.
            profile_data: Structured profile dict (optional).

        Returns:
            Dict with 'analyst_output' (markdown) and 'insights' (list[dict]).
        """
        try:
            insights = self._generate(profiler_output, sample_text, profile_data)
            insights = self.categorizer.categorize_all(insights)
            markdown = self.formatter.to_markdown(insights)

            logger.info(f"AnalystAgent: {len(insights)} insights generated")

            return {
                "analyst_output": markdown,
                "insights": insights,
            }

        except Exception as e:
            logger.error(f"Error in AnalystAgent.run: {e}", exc_info=True)
            return {
                "analyst_output": f"❌ Error during insight generation: {e}",
                "insights": [],
                "error": str(e),
            }

    def _generate(
        self,
        profiler_output: str,
        sample_text: str,
        profile_data: dict | None = None,
    ) -> list[dict]:
        """Call the LLM to generate raw insights."""
        llm = get_llm(temperature=self.temperature, model=self.model)
        handler = get_langfuse_handler()

        shape = profile_data.get("shape", {}) if profile_data else {}
        missing = profile_data.get("missing_values", {}) if profile_data else {}

        system_prompt = PROMPTS["analyst"]["system"]
        human_prompt = PROMPTS["analyst"]["human"].format(
            profiler_output=profiler_output,
            rows=shape.get("rows", "?"),
            cols=shape.get("cols", "?"),
            missing_values=json.dumps(missing, ensure_ascii=False) if missing else "None",
            sample_text=sample_text,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        config = {}
        if handler:
            config["callbacks"] = [handler]

        response = llm.invoke(messages, config=config)
        return self._parse_response(response.content)

    def _parse_response(self, raw: str) -> list[dict]:
        """Parse LLM response into validated insight dicts."""
        parsed = extract_json(raw)

        if parsed and "insights" in parsed:
            insights = [i for i in parsed["insights"] if validate_insight(i)]
        else:
            logger.warning("JSON parsing failed, attempting markdown fallback")
            insights = fallback_parse_markdown(raw)

        for insight in insights:
            insight["priority"] = normalize_priority(insight.get("priority", "medium"))

        if not insights:
            logger.error(f"No insights extracted. Raw response:\n{raw[:500]}")

        return insights


# ═══════════════════════════════════════════════════════════════════════════
#  LANGGRAPH NODE (pipeline entry point)
# ═══════════════════════════════════════════════════════════════════════════


def analyst_node(state: dict) -> dict:
    """
    LangGraph node — Phase 2: generate and structure insights.

    Reads from state:
        - profiler_output (str)
        - sample_text (str)
        - profile_data (dict, optional)

    Writes to state:
        - analyst_output (str): formatted markdown
        - insights (list[dict]): structured insights
    """
    profiler_output = state.get("profiler_output", "")
    sample_text = state.get("sample_text", "")
    profile_data = state.get("profile_data")

    if not profiler_output and not sample_text:
        return {
            "analyst_output": "⚠️ No input data for analysis.",
            "insights": [],
            "error": "No profiler_output or sample_text provided",
        }

    agent = AnalystAgent()
    return agent.run(
        profiler_output=profiler_output,
        sample_text=sample_text,
        profile_data=profile_data,
    )
