# Uncertainty Estimator — Design Document (v2)

## 1. What is the Uncertainty Estimator?

The Uncertainty Estimator assigns a **confidence score (0–100%)** to each
insight. It answers the question: **"How much should we trust this insight?"**

It uses a **hybrid approach**: simple rules for what can be measured
directly (data quality, specificity), and a single LLM call for what
requires understanding (statistical evidence, critique severity).

---

## 2. Why Do We Need It?

Without confidence scores, all insights look equally valid:

```
Insight 1: Sales are concentrated in Ile-de-France       ← is this certain?
Insight 2: There's a correlation between price and qty   ← or is this a guess?
Insight 3: November has a sales spike                    ← based on how much data?
```

With confidence scores:

```
Insight 1: Sales concentrated in Ile-de-France    → 88% confidence ✅
   Reason: 1500 orders, clear %, critic agrees

Insight 2: Price-quantity correlation              → 45% confidence ⚠️
   Reason: No correlation test, only inferred from sample

Insight 3: November sales spike                   → 30% confidence ❌
   Reason: Only 12 data points, high variance, critic says "weak"
```

Now the decision-maker knows what to act on and what to investigate further.

---

## 3. How Does It Compute the Score?

### Hybrid Approach: Rules + LLM

```
┌─────────────────────────────────────────────────┐
│              UNCERTAINTY ESTIMATOR               │
│                                                  │
│   RULE-BASED (no LLM, fast)        50 pts max   │
│   ├── Data Quality Score       → 0–25 points    │
│   └── Specificity Score        → 0–25 points    │
│                                                  │
│   LLM-BASED (1 call per insight)   50 pts max   │
│   ├── Statistical Evidence     → 0–25 points    │
│   └── Critic Assessment        → 0–25 points    │
│                                                  │
│   TOTAL = rules + LLM         → 0–100%          │
└─────────────────────────────────────────────────┘
```

### 3.1 Data Quality Score — Rule-based (0–25 points)

Computed directly from `profile_data`, no LLM needed:

| Factor                        | Good (full points)  | Bad (low points)     |
|-------------------------------|---------------------|----------------------|
| Missing values in key columns | < 1% missing        | > 10% missing        |
| Duplicate rows                | 0 duplicates        | > 5% duplicates      |
| Sample size                   | > 1000 rows         | < 100 rows           |
| Column referenced exists      | Column is real      | Column not in dataset|

**How it works:** Read `profile_data["missing_values"]`,
`profile_data["shape"]["rows"]`, `profile_data["duplicates"]`.
Pure Python, no LLM.

### 3.2 Specificity Score — Rule-based (0–25 points)

Computed by string matching, no LLM needed:

| Factor                        | Good                | Bad                      |
|-------------------------------|---------------------|--------------------------|
| References specific columns   | "column X shows..." | "the data suggests..."   |
| Contains concrete numbers     | "32%", "185 orders" | "a lot", "significant"   |
| Actionable recommendation     | "Expand to region Y"| "Investigate further"    |

**How it works:** Check if insight text contains column names from
`profile_data["columns"]`. Count numeric patterns with regex.
Pure Python, no LLM.

### 3.3 Statistical Evidence Score — LLM-based (0–25 points)

**Why LLM?** Judging whether "32% concentration" is statistically
meaningful requires reasoning that rules can't do. Is a std > mean
alarming? Depends on the distribution type. Only an LLM can assess this.

The LLM receives the insight + profile_data and answers:
```json
{
  "score": 20,
  "reason": "Clear percentage with large effect size, but no statistical test performed"
}
```

### 3.4 Critic Assessment Score — LLM-based (0–25 points)

**Why LLM?** The Critic's verdict is `"partially_supported"` — we could
map that to 15 points with a rule. But the Critic also writes text
explaining *why*. Understanding whether the weaknesses are minor
("missing 0.3% values") or major ("confounding variable invalidates
the entire insight") requires comprehension. The LLM reads the
critique text and scores accordingly.

The LLM receives the insight + critique and answers:
```json
{
  "score": 12,
  "reason": "Critic identifies a significant confounding factor (population density) that partially undermines the insight"
}
```

### 3.5 Combined into One LLM Call

To keep it efficient, both LLM scores come from **a single prompt**:

```
Given this insight, the dataset profile, and the critic's review,
score two dimensions (0-25 each):

1. Statistical Evidence: How well is this insight supported by data?
2. Critic Assessment: How severe are the weaknesses the critic found?

Reply ONLY with JSON:
{
  "statistical_evidence": {"score": ..., "reason": "..."},
  "critic_assessment": {"score": ..., "reason": "..."}
}
```

**One LLM call per insight. That's it.**

### Final Score

```
confidence = data_quality + specificity + statistical_evidence + critic_assessment
           =    22        +     22      +       20             +       12
           =    76%

→ "Medium confidence"
```

---

## 4. Confidence Levels

| Score    | Level  | Icon | Meaning                                      |
|----------|--------|------|----------------------------------------------|
| 80–100%  | High   | 🟢   | Act on this insight with confidence           |
| 50–79%   | Medium | 🟡   | Useful but verify before major decisions      |
| 0–49%    | Low    | 🔴   | Treat as hypothesis, needs more investigation |

---

## 5. Position in the Architecture

```
AnalystAgent
     │
     ▼
  insights (list[dict])
     │
     ▼
CriticAgent   ← reads insights + profile_data
     │
     ▼
  critiques (list[dict])    ← contains verdict, weaknesses, etc.
     │
     ▼
UncertaintyEstimator   ← reads insights + critiques + profile_data
     │                     Rules: data_quality + specificity (no LLM)
     │                     LLM:   statistical_evidence + critic_assessment
     │
     ▼
  confidence_scores (list[dict])
     │
     ▼
ReporterAgent   ← receives insights + critiques + confidence_scores
     │
     ▼
  Enhanced Report
```

**Dependency chain:** Analyst → Critic → Uncertainty → Reporter

The Uncertainty Estimator CANNOT run before the Critic — it needs the
Critic's verdict and weaknesses as input.

---

## 6. What It Produces

For each insight:

```json
{
  "insight_title": "Sales concentrated in Ile-de-France",
  "confidence_score": 76,
  "confidence_level": "medium",
  "drivers": {
    "data_quality": {
      "score": 22,
      "max": 25,
      "reason": "1500 rows, 0.8% missing in relevant columns, 3 duplicates"
    },
    "specificity": {
      "score": 22,
      "max": 25,
      "reason": "References 'region' column, cites exact count (480 orders, 32%)"
    },
    "statistical_evidence": {
      "score": 20,
      "max": 25,
      "reason": "Clear percentage with large effect size, but no statistical test"
    },
    "critic_assessment": {
      "score": 12,
      "max": 25,
      "reason": "Critic identifies population density as confounding factor"
    }
  },
  "summary": "Medium confidence (76%). Well-supported by data volume and specificity, but the Critic raises a valid concern about confounding factors."
}
```

---

## 7. Critic Agent vs Uncertainty Estimator — Final Comparison

| Aspect            | Critic Agent                        | Uncertainty Estimator              |
|-------------------|-------------------------------------|------------------------------------|
| **Question**      | "Is this reasoning sound?"          | "How confident should we be?"      |
| **Output**        | Text (strengths, weaknesses, etc.)  | Number (0–100%) + drivers          |
| **Method**        | LLM only (reviews reasoning)        | Hybrid: rules + LLM               |
| **Uses LLM?**     | Yes (1 call per insight)            | Yes (1 call per insight for 2 scores) |
| **Input**         | insight + profile_data              | insight + critique + profile_data  |
| **Depends on**    | Analyst only                        | Analyst + Critic                   |
| **In report**     | "However, this may be because..."   | "Confidence: 76% 🟡"              |
| **Analogy**       | Peer reviewer writing comments      | Risk score on a dashboard          |

---

## 8. What the Final Report Looks Like

```markdown
## Key Insights

### 🟡 Sales concentrated in Ile-de-France (Confidence: 76%)

**Observation:** 32% of orders come from this region (480 out of 1500).

**Critique:** While the concentration is real, Ile-de-France represents
19% of France's population. The over-representation is notable but less
extreme than the raw number suggests.

**Confidence breakdown:**
- Data quality: 22/25 — large dataset, minimal missing values
- Specificity: 22/25 — references exact column and numbers
- Statistical evidence: 20/25 — clear effect, no formal test
- Critic assessment: 12/25 — valid confounding factor identified

**Recommendation:** Normalize sales by regional population before
deciding on expansion strategy.
```

---

## 9. Implementation Plan

```
agents/uncertainty.py
├── UncertaintyEstimator (class)
│   ├── estimate(insight, critique, profile_data) → dict
│   ├── estimate_all(insights, critiques, profile_data) → list[dict]
│   ├── _compute_data_quality(profile_data) → int        (rule-based)
│   ├── _compute_specificity(insight, profile_data) → int (rule-based)
│   ├── _compute_llm_scores(insight, critique) → dict     (LLM-based)
│   └── _determine_level(score) → str
├── uncertainty_node(state) → dict   (LangGraph entry point)
└── UncertaintyFormatter (class)
    └── to_markdown(scores) → str

config/prompts.yaml
└── uncertainty:
    ├── system: ...
    └── human: ...

tests/test_uncertainty.py
```
