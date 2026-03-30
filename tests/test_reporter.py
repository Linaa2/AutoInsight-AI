"""
tests/test_reporter.py — Reporter Agent tests.

Run from the project root:
    uv run python tests/test_reporter.py
"""

import sys
from pathlib import Path

# Fix import: add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import pytest

from agents.mock_profiler import get_mock_state
from agents.reporter import (
    ReporterAgent,
    build_charts_summary,
    build_insights_summary,
    extract_chart_titles,
    reporter_node,
)

# ═══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS (no LLM)
# ═══════════════════════════════════════════════════════════════════════════


def test_extract_chart_titles():
    """Test chart title extraction from visualizer output."""
    viz = {
        "charts": [
            {"title": "Sales by Region", "type": "bar", "code": "..."},
            {"title": "Price Distribution", "type": "histogram", "code": "..."},
            {"title": "Monthly Trend", "type": "line", "code": "..."},
        ]
    }
    titles = extract_chart_titles(viz)
    assert len(titles) == 3
    assert titles[0] == "Sales by Region"
    print("✅ extract_chart_titles: 3 titles extracted")


def test_extract_chart_titles_empty():
    """Test chart title extraction with no charts."""
    assert extract_chart_titles(None) == []
    assert extract_chart_titles({}) == []
    assert extract_chart_titles({"charts": []}) == []
    print("✅ extract_chart_titles (empty): OK")


def test_build_charts_summary():
    """Test charts summary formatting."""
    viz = {
        "charts": [
            {"title": "Sales by Region", "type": "bar"},
            {"title": "Price Distribution", "type": "histogram"},
        ]
    }
    summary = build_charts_summary(viz)
    assert "Sales by Region" in summary
    assert "Price Distribution" in summary
    assert summary.startswith("- Chart:")
    print("✅ build_charts_summary: formatted correctly")


def test_build_charts_summary_none():
    """Test charts summary with no visualizer output."""
    summary = build_charts_summary(None)
    assert "No charts" in summary
    print("✅ build_charts_summary (none): OK")


def test_build_insights_summary():
    """Test insights summary formatting."""
    insights = [
        {
            "title": "High concentration in Ile-de-France",
            "observation": "32% of orders",
            "recommendation": "Expand to other regions",
            "priority": "high",
            "category": "distribution",
        },
        {
            "title": "Widget Pro dominates",
            "observation": "12.3% of sales",
            "recommendation": "Analyze margin",
            "priority": "medium",
            "category": "general",
        },
    ]
    summary = build_insights_summary(insights)
    assert "High" in summary
    assert "Distribution" in summary
    assert "Widget Pro" in summary
    assert "1." in summary
    assert "2." in summary
    print("✅ build_insights_summary: formatted correctly")


def test_build_insights_summary_empty():
    """Test insights summary with no insights."""
    summary = build_insights_summary(None)
    assert "No structured insights" in summary
    summary2 = build_insights_summary([])
    assert "No structured insights" in summary2
    print("✅ build_insights_summary (empty): OK")


# ═══════════════════════════════════════════════════════════════════════════
#  LLM TESTS
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_reporter_agent_with_llm():
    """Test the full ReporterAgent pipeline with LLM."""
    from utils.llm import LLMClient

    print("\n" + "=" * 60)
    print("  TEST REPORTER WITH LLM")
    print("=" * 60)

    print("\n🔌 Checking LLM availability...")
    try:
        llm = LLMClient.get_text_llm()
        llm.invoke("ping")
        print("   → LLM connected ✓")
    except Exception as e:
        print(f"❌ LLM not reachable: {e}")
        print("   → Run 'ollama serve' in another terminal")
        return False

    # Build mock inputs
    state = get_mock_state()
    mock_analyst_output = (
        "# Insights\n"
        "## 📈 Trend\n"
        "### 🔴 Sales concentrated in Ile-de-France\n"
        "32% of orders come from one region.\n"
        "## ⚠️ Anomaly\n"
        "### 🟡 Extreme order value detected\n"
        "Max total of 2499.75 with mean of 159.68.\n"
    )
    mock_insights = [
        {
            "title": "Sales concentrated in Ile-de-France",
            "observation": "32% of orders from one region",
            "hypothesis": "Population density",
            "recommendation": "Expand to other regions",
            "priority": "high",
            "category": "distribution",
        },
        {
            "title": "Extreme order value detected",
            "observation": "Max total 2499.75 vs mean 159.68",
            "hypothesis": "Bulk order or error",
            "recommendation": "Investigate outliers",
            "priority": "medium",
            "category": "anomaly",
        },
    ]
    mock_viz = {
        "charts": [
            {"title": "Sales by Region", "type": "bar"},
            {"title": "Order Value Distribution", "type": "histogram"},
        ]
    }

    print("\n🤖 Running ReporterAgent...")
    print("   (15-60 seconds depending on your machine)\n")

    agent = ReporterAgent()
    result = agent.run(
        profiler_output=state["profiler_output"],
        analyst_output=mock_analyst_output,
        insights=mock_insights,
        visualizer_output=mock_viz,
    )

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return False

    report = result.get("reporter_output", "")
    if len(report) < 100:
        print(f"❌ Report too short ({len(report)} chars)")
        return False

    print(f"✅ Report generated ({len(report)} chars)\n")
    print("-" * 60)
    print(report[:1000])
    if len(report) > 1000:
        print(f"\n... ({len(report) - 1000} more chars)")
    print("-" * 60)

    return True


@pytest.mark.integration
def test_reporter_node_with_llm():
    """Test the LangGraph node entry point."""
    from utils.llm import LLMClient

    try:
        LLMClient.get_text_llm().invoke("ping")
    except Exception:
        print("⏭️  LLM unavailable, skipping test_reporter_node_with_llm")
        return False

    print("\n🔗 Testing reporter_node() (LangGraph entry point)...")

    state = get_mock_state()
    state["analyst_output"] = "Some analyst insights in markdown."
    state["insights"] = [
        {
            "title": "Test insight",
            "observation": "Obs",
            "hypothesis": "Hyp",
            "recommendation": "Rec",
            "priority": "high",
            "category": "general",
        }
    ]

    result = reporter_node(state)

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return False

    output = result.get("reporter_output", "")
    assert len(output) > 50, "Report too short"

    print(f"✅ reporter_node: report of {len(output)} chars")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  UNIT TESTS (no LLM)")
    print("=" * 60)
    print()

    test_extract_chart_titles()
    test_extract_chart_titles_empty()
    test_build_charts_summary()
    test_build_charts_summary_none()
    test_build_insights_summary()
    test_build_insights_summary_empty()

    print("\n✅ All unit tests passed!\n")

    # LLM tests
    ok1 = test_reporter_agent_with_llm()
    if ok1:
        ok2 = test_reporter_node_with_llm()

    print("\n" + "=" * 60)
    if ok1:
        print("  ✅ REPORTER OPERATIONAL — ReporterAgent works!")
    else:
        print("  ⚠️  Unit tests OK, but LLM test failed.")
        print("  Check that Ollama is running: ollama serve")
    print("=" * 60)
