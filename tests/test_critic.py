"""
tests/test_critic.py — Critic Agent tests.
"""

import pytest

from agents.critic import (
    CriticAgent,
    CriticFormatter,
    critic_node,
    normalize_confidence,
    normalize_verdict,
    validate_critique,
)
from agents.mock_profiler import get_mock_state

# ═══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS (no LLM)
# ═══════════════════════════════════════════════════════════════════════════


def test_validate_critique():
    """Test critique validation."""
    good = {
        "strengths": "Well supported",
        "weaknesses": "Some gaps",
        "alternatives": "Could be X",
        "confidence": "high",
        "verdict": "supported",
    }
    bad = {"strengths": "OK", "weaknesses": ""}

    assert validate_critique(good) is True
    assert validate_critique(bad) is False


def test_validate_critique_missing_field():
    """Test critique with missing fields."""
    missing_verdict = {
        "strengths": "Good",
        "weaknesses": "Bad",
        "alternatives": "Maybe",
        "confidence": "high",
    }
    assert validate_critique(missing_verdict) is False


def test_normalize_verdict():
    """Test verdict normalization."""
    assert normalize_verdict("supported") == "supported"
    assert normalize_verdict("strong") == "supported"
    assert normalize_verdict("valid") == "supported"
    assert normalize_verdict("weak") == "weak"
    assert normalize_verdict("unsupported") == "weak"
    assert normalize_verdict("partially supported") == "partially_supported"
    assert normalize_verdict("partially_supported") == "partially_supported"
    assert normalize_verdict("random text") == "partially_supported"


def test_normalize_confidence():
    """Test confidence normalization."""
    assert normalize_confidence("high") == "high"
    assert normalize_confidence("strong") == "high"
    assert normalize_confidence("low") == "low"
    assert normalize_confidence("weak") == "low"
    assert normalize_confidence("faible") == "low"
    assert normalize_confidence("medium") == "medium"
    assert normalize_confidence("whatever") == "medium"


def test_formatter():
    """Test CriticFormatter output."""
    critiques = [
        {
            "insight_title": "Sales in IDF",
            "strengths": "Backed by data",
            "weaknesses": "No population normalization",
            "alternatives": "Marketing spend could explain it",
            "confidence": "medium",
            "verdict": "partially_supported",
        },
        {
            "insight_title": "Widget Pro dominates",
            "strengths": "Clear numbers",
            "weaknesses": "No margin data",
            "alternatives": "Could be due to pricing",
            "confidence": "high",
            "verdict": "supported",
        },
    ]
    md = CriticFormatter.to_markdown(critiques)
    assert "✅" in md
    assert "⚠️" in md
    assert "Sales in IDF" in md
    assert "Widget Pro" in md
    assert "Summary" in md


def test_formatter_empty():
    """Test CriticFormatter with empty list."""
    md = CriticFormatter.to_markdown([])
    assert "No critiques" in md


def test_formatter_all_verdicts():
    """Test CriticFormatter with all verdict types."""
    critiques = [
        {
            "insight_title": "A",
            "strengths": "S",
            "weaknesses": "W",
            "alternatives": "A",
            "confidence": "high",
            "verdict": "supported",
        },
        {
            "insight_title": "B",
            "strengths": "S",
            "weaknesses": "W",
            "alternatives": "A",
            "confidence": "medium",
            "verdict": "partially_supported",
        },
        {
            "insight_title": "C",
            "strengths": "S",
            "weaknesses": "W",
            "alternatives": "A",
            "confidence": "low",
            "verdict": "weak",
        },
    ]
    md = CriticFormatter.to_markdown(critiques)
    assert "✅ 1 supported" in md
    assert "⚠️ 1 partially supported" in md
    assert "❌ 1 weak" in md


# ═══════════════════════════════════════════════════════════════════════════
#  LLM INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_critic_agent_with_llm():
    """Test full CriticAgent pipeline with LLM."""
    from utils.llm import LLMClient

    LLMClient.get_text_llm().invoke("ping")

    state = get_mock_state()
    mock_insights = [
        {
            "title": "Sales concentrated in Ile-de-France",
            "observation": "32% of orders (480 out of 1500) come from Ile-de-France",
            "hypothesis": "Population density and logistics proximity",
            "recommendation": "Expand to other regions",
            "priority": "high",
            "category": "distribution",
        },
        {
            "title": "High dispersion in order totals",
            "observation": "Standard deviation (185.42) exceeds the mean (159.68)",
            "hypothesis": "Mix of small and bulk orders",
            "recommendation": "Segment orders by size",
            "priority": "medium",
            "category": "anomaly",
        },
    ]

    critic = CriticAgent()
    result = critic.run(
        insights=mock_insights,
        profile_data=state["profile_data"],
    )

    assert not result.get("error"), f"Critic error: {result.get('error')}"
    critiques = result.get("critiques", [])
    assert len(critiques) > 0, "No critiques generated"


@pytest.mark.integration
def test_critic_node_with_llm():
    """Test the LangGraph node entry point."""
    from utils.llm import LLMClient

    LLMClient.get_text_llm().invoke("ping")

    state = get_mock_state()
    state["insights"] = [
        {
            "title": "Test insight",
            "observation": "Some observation about the data",
            "hypothesis": "A possible explanation",
            "recommendation": "Do something about it",
            "priority": "high",
            "category": "general",
        },
    ]

    result = critic_node(state)

    assert not result.get("error"), f"Error: {result.get('error')}"
    critiques = result.get("critiques", [])
    output = result.get("critic_output", "")
    assert len(critiques) > 0, "No critiques in result"
    assert len(output) > 50, "Markdown too short"


def test_critic_node_no_insights():
    """Test critic_node with empty insights."""
    result = critic_node({"insights": []})
    assert result.get("error") is not None
    assert len(result.get("critiques", [])) == 0


def test_critic_batches_multiple_insights(monkeypatch):
    """Multiple insights should be critiqued in a single batch LLM call."""
    calls: list[str] = []

    def fake_call(*, system, human, task, **kwargs):  # noqa: ARG001
        calls.append(human)
        assert task == "critic"
        return """
        {
          "critiques": [
            {
              "index": 0,
              "insight_title": "Insight A",
              "strengths": "Well-supported.",
              "weaknesses": "Some caveats.",
              "alternatives": "Could be seasonality.",
              "confidence": "high",
              "verdict": "supported"
            },
            {
              "index": 1,
              "insight_title": "Insight B",
              "strengths": "Reasonable signal.",
              "weaknesses": "Small sample.",
              "alternatives": "Could be noise.",
              "confidence": "medium",
              "verdict": "partially_supported"
            }
          ]
        }
        """

    monkeypatch.setattr("agents.critic.call_llm_with_messages", fake_call)

    insights = [
        {
            "title": "Insight A",
            "observation": "Observation A",
            "hypothesis": "Hypothesis A",
            "recommendation": "Recommendation A",
            "priority": "high",
        },
        {
            "title": "Insight B",
            "observation": "Observation B",
            "hypothesis": "Hypothesis B",
            "recommendation": "Recommendation B",
            "priority": "medium",
        },
    ]

    result = CriticAgent().run(insights=insights, profile_data=get_mock_state()["profile_data"])

    assert len(calls) == 1
    assert len(result["critiques"]) == 2
    assert result["critiques"][0]["insight_title"] == "Insight A"
    assert result["critiques"][1]["insight_title"] == "Insight B"


def test_critic_batch_falls_back_to_single_for_missing_items(monkeypatch):
    """Missing batch results should fall back to the single-insight path."""
    calls: list[str] = []

    def fake_call(*, system, human, task, **kwargs):  # noqa: ARG001
        calls.append(human)
        if "### Insight 1" in human:
            return """
            {
              "critiques": [
                {
                  "index": 0,
                  "insight_title": "Insight A",
                  "strengths": "Well-supported.",
                  "weaknesses": "Some caveats.",
                  "alternatives": "Could be seasonality.",
                  "confidence": "high",
                  "verdict": "supported"
                }
              ]
            }
            """

        return """
        {
          "strengths": "Reasonable signal.",
          "weaknesses": "Small sample.",
          "alternatives": "Could be noise.",
          "confidence": "medium",
          "verdict": "partially_supported"
        }
        """

    monkeypatch.setattr("agents.critic.call_llm_with_messages", fake_call)

    insights = [
        {
            "title": "Insight A",
            "observation": "Observation A",
            "hypothesis": "Hypothesis A",
            "recommendation": "Recommendation A",
            "priority": "high",
        },
        {
            "title": "Insight B",
            "observation": "Observation B",
            "hypothesis": "Hypothesis B",
            "recommendation": "Recommendation B",
            "priority": "medium",
        },
    ]

    result = CriticAgent().run(insights=insights, profile_data=get_mock_state()["profile_data"])

    assert len(calls) == 2
    assert len(result["critiques"]) == 2
    assert result["critiques"][1]["insight_title"] == "Insight B"
