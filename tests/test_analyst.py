"""
tests/test_analyst.py — Phase 2 tests (Analyst Agent).

Run from the project root:
    uv run python tests/test_analyst.py
"""

import sys
from pathlib import Path

# Fix import: add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.analyst import (
    AnalystAgent,
    InsightCategorizer,
    InsightFormatter,
    analyst_node,
    extract_json,
    fallback_parse_markdown,
    normalize_priority,
    validate_insight,
)
from agents.mock_profiler import get_mock_state

# ═══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS (parsing, categorization, formatting — no LLM)
# ═══════════════════════════════════════════════════════════════════════════


def test_extract_json():
    """Test JSON extraction from a noisy LLM response."""
    raw = """
    Here are the insights:
    ```json
    {
      "insights": [
        {
          "title": "High concentration in Ile-de-France",
          "observation": "32% of orders come from Ile-de-France",
          "hypothesis": "Population density and logistics proximity",
          "recommendation": "Expand into other regions",
          "priority": "high"
        },
        {
          "title": "Widget Pro dominates sales",
          "observation": "185 orders out of 1500, i.e. 12.3%",
          "hypothesis": "Good value for money",
          "recommendation": "Analyze the margin on this product",
          "priority": "medium"
        }
      ]
    }
    ```
    """
    parsed = extract_json(raw)
    assert parsed is not None, "JSON extraction failed"
    assert len(parsed["insights"]) == 2
    print("✅ extract_json: 2 insights extracted")


def test_validate_insight():
    """Test required field validation."""
    good = {
        "title": "Test",
        "observation": "Obs",
        "hypothesis": "Hyp",
        "recommendation": "Rec",
        "priority": "high",
    }
    bad_missing = {"title": "Test", "observation": ""}
    bad_empty = {
        "title": "Test",
        "observation": "Obs",
        "hypothesis": "",
        "recommendation": "Rec",
        "priority": "high",
    }

    assert validate_insight(good) is True
    assert validate_insight(bad_missing) is False
    assert validate_insight(bad_empty) is False
    print("✅ validate_insight: validation OK")


def test_normalize_priority():
    """Test priority normalization."""
    assert normalize_priority("high") == "high"
    assert normalize_priority("haute") == "high"
    assert normalize_priority("élevée") == "high"
    assert normalize_priority("low") == "low"
    assert normalize_priority("basse") == "low"
    assert normalize_priority("medium") == "medium"
    assert normalize_priority("random") == "medium"
    print("✅ normalize_priority: normalization OK")


def test_categorizer_keywords():
    """Test InsightCategorizer with keyword mode."""
    categorizer = InsightCategorizer(use_llm=False)

    trend = {
        "title": "Sales rise in Q4",
        "observation": "30% increase in December",
        "hypothesis": "Seasonal effect",
    }
    anomaly = {
        "title": "Unusual order detected",
        "observation": "An outlier at 2499€",
        "hypothesis": "Error or exceptional order",
    }
    corr = {
        "title": "Price-quantity correlation",
        "observation": "Inversely proportional relationship",
        "hypothesis": "Price elasticity",
    }

    assert categorizer.categorize(trend) == "trend"
    assert categorizer.categorize(anomaly) == "anomaly"
    assert categorizer.categorize(corr) == "correlation"
    print("✅ InsightCategorizer (keywords): categorization OK")


def test_categorizer_batch():
    """Test InsightCategorizer.categorize_all."""
    categorizer = InsightCategorizer(use_llm=False)

    insights = [
        {"title": "Sales trend up", "observation": "Increase monthly", "hypothesis": "Growth"},
        {"title": "Outlier detected", "observation": "Extreme spike", "hypothesis": "Error"},
    ]
    result = categorizer.categorize_all(insights)

    assert result[0]["category"] == "trend"
    assert result[1]["category"] == "anomaly"
    print("✅ InsightCategorizer.categorize_all: batch OK")


def test_fallback_parse_markdown():
    """Test insight extraction from markdown (fallback)."""
    md = """\
**Title**: Sales are increasing
**Observation**: +15% over the quarter
**Hypothesis**: Marketing campaign
**Recommendation**: Continue the effort
**Priority**: high

**Title**: Low stock
**Observation**: Out of stock on 3 products
**Hypothesis**: Underestimated demand
**Recommendation**: Adjust forecasts
**Priority**: medium
"""
    result = fallback_parse_markdown(md)
    assert len(result) == 2
    assert result[0]["title"] == "Sales are increasing"
    assert result[1]["title"] == "Low stock"
    print("✅ fallback_parse_markdown: 2 insights extracted from markdown")


def test_formatter():
    """Test InsightFormatter.to_markdown."""
    insights = [
        {
            "title": "Test trend",
            "observation": "Obs",
            "hypothesis": "Hyp",
            "recommendation": "Rec",
            "priority": "high",
            "category": "trend",
        },
        {
            "title": "Test anomaly",
            "observation": "Obs",
            "hypothesis": "Hyp",
            "recommendation": "Rec",
            "priority": "low",
            "category": "anomaly",
        },
    ]
    md = InsightFormatter.to_markdown(insights)
    assert "📈 Trend" in md
    assert "⚠️ Anomaly" in md
    assert "🔴" in md  # high
    assert "🟢" in md  # low
    print("✅ InsightFormatter.to_markdown: formatting OK")


def test_formatter_empty():
    """Test InsightFormatter with empty list."""
    md = InsightFormatter.to_markdown([])
    assert "No insights" in md
    print("✅ InsightFormatter.to_markdown (empty): OK")


# ═══════════════════════════════════════════════════════════════════════════
#  LLM TESTS (Ollama + Mistral)
# ═══════════════════════════════════════════════════════════════════════════


def test_analyst_agent_with_llm():
    """Test the full AnalystAgent pipeline with Ollama."""
    from utils.llm import check_ollama_health

    print("\n" + "=" * 60)
    print("  TEST WITH OLLAMA (Mistral)")
    print("=" * 60)

    print("\n🔌 Checking Ollama...")
    if not check_ollama_health():
        print("❌ Ollama is not reachable.")
        print("   → Run 'ollama serve' in another terminal")
        print("   → Then verify with 'ollama list'")
        return False

    print("   → Ollama connected ✓")

    state = get_mock_state()
    print("📦 Mock state loaded")

    print("\n🤖 Running AnalystAgent...")
    print("   (15-60 seconds depending on your machine)\n")

    agent = AnalystAgent(temperature=0.5)
    result = agent.run(
        profiler_output=state["profiler_output"],
        sample_text=state["sample_text"],
        profile_data=state["profile_data"],
    )

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return False

    insights = result.get("insights", [])
    if not insights:
        print("❌ No insights generated. The LLM did not return a usable format.")
        return False

    print(f"✅ {len(insights)} insights generated!\n")

    for i, ins in enumerate(insights, 1):
        print(
            f"  [{i}] {ins.get('category', '?').upper():15} | "
            f"{ins['priority'].upper():6} | {ins['title']}"
        )

    print("\n📝 Markdown output:")
    print("-" * 60)
    print(result["analyst_output"])
    print("-" * 60)

    return True


def test_analyst_node_with_llm():
    """Test the LangGraph node entry point."""
    from utils.llm import check_ollama_health

    if not check_ollama_health():
        print("⏭️  Ollama unavailable, skipping test_analyst_node_with_llm")
        return False

    print("\n🔗 Testing analyst_node() (LangGraph entry point)...")

    state = get_mock_state()
    result = analyst_node(state)

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return False

    insights = result.get("insights", [])
    output = result.get("analyst_output", "")

    assert len(insights) > 0, "No insights in result"
    assert len(output) > 50, "Markdown too short"

    print(f"✅ analyst_node: {len(insights)} insights, {len(output)} chars of markdown")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  UNIT TESTS (no LLM)")
    print("=" * 60)
    print()

    test_extract_json()
    test_validate_insight()
    test_normalize_priority()
    test_categorizer_keywords()
    test_categorizer_batch()
    test_fallback_parse_markdown()
    test_formatter()
    test_formatter_empty()

    print("\n✅ All unit tests passed!\n")

    # LLM tests
    ok1 = test_analyst_agent_with_llm()
    if ok1:
        ok2 = test_analyst_node_with_llm()

    print("\n" + "=" * 60)
    if ok1:
        print("  ✅ PHASE 2 OPERATIONAL — Analyst agent works!")
    else:
        print("  ⚠️  Unit tests OK, but LLM test failed.")
        print("  Check that Ollama is running: ollama serve")
    print("=" * 60)
