# Phase 8 — Critique & Uncertainty System — Design Document

## 1. Overview

Phase 8 adds an **intelligence layer** to the pipeline: two agents that work
together to make insights more trustworthy.

```
         ┌─────────────┐
         │   Analyst    │  Generates insights
         └──────┬──────┘
                │
                │  insights (list[dict])
                │  Each insight has: title, observation, hypothesis,
                │                    recommendation, priority, category
                │
                ▼
         ┌─────────────┐
         │   Critic     │  Challenges each insight
         │   Agent      │  1 LLM call per insight
         └──────┬──────┘
                │
                │  critiques (list[dict])
                │  Each critique has: insight_title, strengths, weaknesses,
                │                     alternatives, confidence, verdict
                │
                ▼
         ┌─────────────┐
         │ Uncertainty  │  Scores each insight
         │ Estimator    │  Rules + 1 LLM call per insight
         └──────┬──────┘
                │
                │  confidence_scores (list[dict])
                │  Each score has: insight_title, confidence_score (0-100%),
                │                  confidence_level, drivers, summary
                │
                ▼
         ┌─────────────┐
         │  Reporter    │  Writes the final report
         │              │  Now receives: insights + critiques + scores
         └─────────────┘
```

**Key rule:** The Critic and Uncertainty Estimator are coded separately but
designed to work together. The contract between them is defined in Section 5.

---

## 2. Agent 1 — Critic Agent

### 2.1 What It Does

The Critic Agent is an **adversarial reviewer**. For each insight from the
Analyst, it asks:
- Is this observation well-supported by the data?
- Are there logical gaps or unsupported assumptions?
- What alternative explanations did the Analyst miss?

It does NOT replace the Analyst's insights — it adds a critique alongside them.

### 2.2 Position in the Pipeline

```
Analyst → [Critic] → Uncertainty → Reporter
```

- **Reads from state:** `insights`, `profiler_output`, `profile_data`
- **Writes to state:** `critiques`, `critic_output`
- **Depends on:** Analyst only
- **Used by:** Uncertainty Estimator + Reporter

### 2.3 Input

The Critic receives each insight as a dict:

```json
{
  "title": "Sales concentrated in Ile-de-France",
  "observation": "32% of orders (480/1500) come from this region",
  "hypothesis": "Population density and logistics proximity",
  "recommendation": "Expand to other regions",
  "priority": "high",
  "category": "distribution"
}
```

Plus the `profiler_output` (markdown) and `profile_data` (dict) for
cross-checking.

### 2.4 Output — The Critique Contract

⚠️ **This is the contract the Uncertainty Estimator depends on.**
Every critique MUST have these exact fields:

```json
{
  "insight_title": "Sales concentrated in Ile-de-France",
  "strengths": "Backed by data: 480/1500 orders (32%) from one region.",
  "weaknesses": "Does not account for population density. Ile-de-France has 19% of France's population, so 32% may not be as extreme as presented.",
  "alternatives": "The concentration could reflect marketing spend or warehouse location rather than real demand.",
  "confidence": "medium",
  "verdict": "partially_supported"
}
```

| Field           | Type   | Values                                         | Used by Uncertainty? |
|-----------------|--------|------------------------------------------------|----------------------|
| `insight_title` | str    | Must match the Analyst's insight title          | ✅ Yes (for mapping)  |
| `strengths`     | str    | What is well-supported (1–3 sentences)          | ❌ No                |
| `weaknesses`    | str    | Logical gaps, missing context (1–3 sentences)   | ✅ Yes (LLM reads it) |
| `alternatives`  | str    | Alternative hypotheses (1–3 sentences)          | ✅ Yes (LLM reads it) |
| `confidence`    | str    | `"high"` / `"medium"` / `"low"`                | ✅ Yes (as fallback)  |
| `verdict`       | str    | `"supported"` / `"partially_supported"` / `"weak"` | ✅ Yes (as fallback) |

**Why `insight_title` must match:** The Uncertainty Estimator maps each
critique to its corresponding insight by matching `insight_title` with the
Analyst's `title`. If they don't match, the Estimator can't find the critique
and will use default scores.

### 2.5 LLM Usage

- **1 LLM call per insight** using the text model
- Prompt asks for JSON response with the 5 fields above
- Fallback: if LLM response can't be parsed, the insight gets no critique
  (Uncertainty Estimator will use default scores)

### 2.6 Implementation

```
agents/critic.py
├── CriticAgent (class)
│   ├── run(insights, profiler_output, profile_data) → dict
│   │     Returns: {"critiques": list[dict], "critic_output": str}
│   ├── _critique_single(insight, profiler_output, profile_data) → dict | None
│   ├── _parse_response(raw) → dict | None
│   └── _build_profile_summary(profile_data) → str
├── CriticFormatter (class)
│   └── to_markdown(critiques) → str
├── critic_node(state) → dict   (LangGraph entry point)
└── Helper functions: validate_critique(), normalize_verdict(), normalize_confidence()
```

---

## 3. Agent 2 — Uncertainty Estimator

### 3.1 What It Does

The Uncertainty Estimator assigns a **confidence score (0–100%)** to each
insight. It answers: "How much should we trust this?"

It uses a **hybrid approach**:
- **Rules** (no LLM): measure data quality and insight specificity
- **LLM** (1 call per insight): evaluate statistical evidence and critique severity

### 3.2 Position in the Pipeline

```
Analyst → Critic → [Uncertainty] → Reporter
```

- **Reads from state:** `insights`, `critiques`, `profile_data`
- **Writes to state:** `confidence_scores`, `uncertainty_output`
- **Depends on:** Analyst + Critic
- **Used by:** Reporter

### 3.3 Input

The Estimator needs three things:

1. **The insight** (from Analyst) — to check specificity and column references
2. **The critique** (from Critic) — to assess severity of weaknesses
3. **The profile_data** (from Profiler) — to check data quality

**How it finds the right critique for each insight:**

```python
# The Estimator builds a lookup map:
critique_map = {}
for critique in critiques:
    critique_map[critique["insight_title"]] = critique

# Then for each insight:
for insight in insights:
    title = insight["title"]
    matching_critique = critique_map.get(title, {})  # empty dict if no match
    score = estimator.estimate(insight, matching_critique, profile_data)
```

**If the Critic failed to critique an insight**, the Estimator still works —
it uses default scores for the critic_assessment driver and falls back to
rule-based scoring only.

### 3.4 How It Computes the Score

```
┌─────────────────────────────────────────────────┐
│              UNCERTAINTY ESTIMATOR               │
│                                                  │
│   RULE-BASED (no LLM, fast)        50 pts max   │
│   ├── Data Quality Score       → 0–25 points    │
│   │   Source: profile_data                       │
│   │   Checks: row count, missing %, duplicates   │
│   │                                              │
│   └── Specificity Score        → 0–25 points    │
│       Source: insight text + profile_data         │
│       Checks: column names, concrete numbers,    │
│               recommendation quality             │
│                                                  │
│   LLM-BASED (1 call per insight)   50 pts max   │
│   ├── Statistical Evidence     → 0–25 points    │
│   │   Source: insight + profile_data             │
│   │   Judges: effect size, statistical support   │
│   │                                              │
│   └── Critic Assessment        → 0–25 points    │
│       Source: insight + critique                  │
│       Judges: severity of weaknesses,            │
│               plausibility of alternatives       │
│                                                  │
│   TOTAL = rules + LLM         → 0–100%          │
└─────────────────────────────────────────────────┘
```

### 3.5 Output — The Confidence Score Contract

⚠️ **This is the contract the Reporter depends on.**

```json
{
  "insight_title": "Sales concentrated in Ile-de-France",
  "confidence_score": 76,
  "confidence_level": "medium",
  "drivers": {
    "data_quality": {
      "score": 22,
      "max": 25,
      "reason": "1500 rows, 0.8% missing, 3 duplicates"
    },
    "specificity": {
      "score": 22,
      "max": 25,
      "reason": "References 'region' column, cites 480 orders (32%)"
    },
    "statistical_evidence": {
      "score": 20,
      "max": 25,
      "reason": "Clear percentage, large effect, no formal test"
    },
    "critic_assessment": {
      "score": 12,
      "max": 25,
      "reason": "Critic identifies population density as confounding factor"
    }
  },
  "summary": "Medium confidence (76%). Strong data support but the Critic raises a valid concern."
}
```

### 3.6 Confidence Levels

| Score    | Level  | Icon | Meaning                                      |
|----------|--------|------|----------------------------------------------|
| 80–100%  | High   | 🟢   | Act on this insight with confidence           |
| 50–79%   | Medium | 🟡   | Useful but verify before major decisions      |
| 0–49%    | Low    | 🔴   | Treat as hypothesis, needs more investigation |

### 3.7 Fallback Behavior

If the Critic didn't produce a critique for a specific insight:

| Driver              | Behavior                                                |
|---------------------|---------------------------------------------------------|
| data_quality        | Normal (uses profile_data, no critique needed)          |
| specificity         | Normal (uses insight text, no critique needed)          |
| statistical_evidence| LLM evaluates without critique context → less accurate  |
| critic_assessment   | Uses default score of 12/25 ("unknown")                 |

If the LLM call fails entirely:

| Driver              | Behavior                                       |
|---------------------|------------------------------------------------|
| statistical_evidence| Default score: 12/25                           |
| critic_assessment   | Maps Critic verdict → score (supported=20, partial=12, weak=5) |

**The Estimator never crashes.** It always returns a score.

### 3.8 Implementation

```
agents/uncertainty.py
├── UncertaintyEstimator (class)
│   ├── estimate_all(insights, critiques, profile_data) → dict
│   │     Returns: {"confidence_scores": list[dict], "uncertainty_output": str}
│   ├── estimate(insight, critique, profile_data) → dict
│   ├── _compute_data_quality(profile_data) → tuple[int, str]     (rule-based)
│   ├── _compute_specificity(insight, profile_data) → tuple[int, str]  (rule-based)
│   ├── _compute_llm_scores(insight, critique) → dict              (LLM-based)
│   └── _determine_level(score) → str
├── UncertaintyFormatter (class)
│   └── to_markdown(scores) → str
├── uncertainty_node(state) → dict   (LangGraph entry point)
└── Helper functions: extract_json()
```

---

## 4. How the Three Evaluation Components Differ

| Component              | Question asked                 | Output           | Method       | Part of pipeline? |
|------------------------|--------------------------------|------------------|--------------|--------------------|
| **LLM-as-Judge**       | "How good is this output?"     | Score (1–5)      | LLM only     | No (post-run)      |
| **Critic Agent**       | "Is the reasoning sound?"      | Structured text  | LLM only     | Yes (in pipeline)  |
| **Uncertainty Estimator** | "How confident should we be?" | Score (0–100%)  | Rules + LLM  | Yes (in pipeline)  |

**Analogy:**
- LLM-as-Judge = **Teacher** grading the exam paper
- Critic Agent = **Peer reviewer** challenging the research
- Uncertainty Estimator = **Risk analyst** assigning probability

---

## 5. The Contract Between Critic and Uncertainty

This is the most important section for team compatibility.

### 5.1 Matching Insights to Critiques

Both agents receive the same `insights` list from the Analyst. The link
between an insight and its critique is the **`title` field**:

```
Analyst insight:    {"title": "Sales concentrated in IDF", ...}
                         ↕  must match
Critic critique:    {"insight_title": "Sales concentrated in IDF", ...}
                         ↕  used for lookup
Uncertainty score:  {"insight_title": "Sales concentrated in IDF", ...}
```

**Rule:** The Critic MUST copy the exact `title` from the insight into
`insight_title` in the critique. No reformulation, no truncation.

### 5.2 State Keys

After Phase 8 runs, the LangGraph state contains:

| Key                  | Type         | Source              | Used by          |
|----------------------|--------------|---------------------|------------------|
| `insights`           | list[dict]   | Analyst             | Critic, Uncertainty, Reporter |
| `critiques`          | list[dict]   | Critic              | Uncertainty, Reporter |
| `critic_output`      | str          | Critic              | Reporter (optional) |
| `confidence_scores`  | list[dict]   | Uncertainty         | Reporter |
| `uncertainty_output` | str          | Uncertainty         | Reporter (optional) |

### 5.3 Developing Separately

The Critic and Uncertainty Estimator can be coded by different people
simultaneously because:

1. **The Critic doesn't need the Uncertainty Estimator** — it only reads
   from the Analyst.

2. **The Uncertainty Estimator can be tested with mock critiques** — just
   create fake critique dicts with the right fields.

3. **Both follow the same code pattern** — `AgentClass` + `Formatter` +
   `node function` + prompts in YAML.

**Mock critique for testing Uncertainty without the Critic:**

```python
mock_critique = {
    "insight_title": "Sales concentrated in Ile-de-France",
    "strengths": "Backed by 480/1500 orders.",
    "weaknesses": "Population density not considered.",
    "alternatives": "Marketing spend could explain it.",
    "confidence": "medium",
    "verdict": "partially_supported",
}
```

**Mock confidence score for testing Reporter without Uncertainty:**

```python
mock_score = {
    "insight_title": "Sales concentrated in Ile-de-France",
    "confidence_score": 76,
    "confidence_level": "medium",
    "drivers": {...},
    "summary": "Medium confidence (76%).",
}
```

---

## 6. LangGraph Integration

When the person handling LangGraph integrates Phase 8, the graph becomes:

```
START → profiler → analyst → critic → uncertainty → visualizer → reporter → END
                                                         ↑
                                          (visualizer runs in parallel
                                           with critic+uncertainty)
```

Or more precisely:

```python
graph.add_edge("analyst", "critic")
graph.add_edge("critic", "uncertainty")
graph.add_edge("analyst", "visualizer")      # parallel branch
graph.add_edge("uncertainty", "reporter")
graph.add_edge("visualizer", "reporter")
```

---

## 7. What the Final Enhanced Report Looks Like

```markdown
## Key Insights

### 🟡 Sales concentrated in Ile-de-France (Confidence: 76%)

**Observation:** 32% of orders come from this region (480 out of 1500).

**Critique:** While the concentration is real, Ile-de-France represents
19% of France's population. The over-representation (32% vs 19%) is
notable but less extreme than the raw number suggests. The concentration
could also reflect marketing spend or warehouse location.

**Confidence breakdown:**
| Driver               | Score  | Detail                                  |
|----------------------|--------|-----------------------------------------|
| Data quality         | 22/25  | Large dataset, minimal missing values   |
| Specificity          | 22/25  | References exact column and numbers     |
| Statistical evidence | 20/25  | Clear effect, no formal test            |
| Critic assessment    | 12/25  | Valid confounding factor identified     |

**Recommendation:** Normalize sales by regional population before
deciding on expansion strategy.
```

---

## 8. Summary

| What              | Critic Agent                           | Uncertainty Estimator                  |
|-------------------|----------------------------------------|----------------------------------------|
| **File**          | `agents/critic.py`                     | `agents/uncertainty.py`                |
| **Class**         | `CriticAgent`                          | `UncertaintyEstimator`                 |
| **LangGraph node**| `critic_node(state)`                   | `uncertainty_node(state)`              |
| **Prompt key**    | `PROMPTS["critic"]`                    | `PROMPTS["uncertainty"]`               |
| **Test file**     | `tests/test_critic_uncertainty.py`     | `tests/test_critic_uncertainty.py`     |
| **Reads**         | insights + profiler_output + profile_data | insights + critiques + profile_data |
| **Writes**        | critiques + critic_output              | confidence_scores + uncertainty_output |
| **LLM calls**     | 1 per insight                          | 1 per insight                          |
| **Can run alone** | ✅ Yes                                  | ⚠️ Yes but uses default scores without Critic |
