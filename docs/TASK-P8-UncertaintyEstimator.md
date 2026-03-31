# P8 — Uncertainty Estimator Agent

## Core Idea

The Uncertainty Estimator assigns a **confidence score (0–100%)** to each analyst insight
by asking: *"How much should we trust this insight?"*

It uses a **hybrid approach** — two drivers are computed by deterministic rules (fast,
no cost), two by a single LLM call (semantic judgment that rules cannot replicate):

```
┌─────────────────────────────────────────────────┐
│              UNCERTAINTY ESTIMATOR               │
│                                                  │
│   RULE-BASED (no LLM, fast)        50 pts max   │
│   ├── Data Quality             → 0–25 points    │
│   │   Source: profile_data                       │
│   │   Checks: row count, missing %, duplicates   │
│   │                                              │
│   └── Specificity              → 0–25 points    │
│       Source: insight text + profile_data        │
│       Checks: column names, concrete numbers,    │
│               recommendation quality             │
│                                                  │
│   LLM-BASED (1 call per insight)   50 pts max   │
│   ├── Statistical Evidence     → 0–25 points    │
│   │   Source: insight + profile_data             │
│   │   Judges: effect size, statistical support   │
│   │                                              │
│   └── Critic Assessment        → 0–25 points    │
│       Source: insight + critique                 │
│       Judges: severity of weaknesses,            │
│               plausibility of alternatives       │
│                                                  │
│   TOTAL = rules + LLM         → 0–100%          │
└─────────────────────────────────────────────────┘
```

---

## How It Differs from the Other Evaluation Components

| Component | Question asked | Output | Method | Pipeline position |
|---|---|---|---|---|
| **LLM-as-Judge (P7)** | How good is this output? | Score 1–5 per rubric | LLM only | Post-run (optional tab) |
| **Critic Agent** | Is the reasoning sound? | Strengths / weaknesses / verdict | LLM only | In pipeline (after Analyst) |
| **Uncertainty Estimator** | How confident should we be? | Score 0–100% + breakdown | Rules + LLM | In pipeline (after Critic) |

**Analogy:**
- LLM-as-Judge = **Teacher** grading the exam paper
- Critic Agent = **Peer reviewer** challenging the research
- Uncertainty Estimator = **Risk analyst** assigning probability

---

## Position in the Pipeline

```
START
  │
  ▼
ProfilerAgent ──► profiler_output (markdown) + profile_data (dict)
  │
  ▼
AnalystAgent ──► insights: list[dict]
  │
  ▼
CriticAgent ──► critiques: list[dict]        ◄── reads: insights + profiler_output + profile_data
  │
  ▼
UncertaintyEstimator ──► confidence_scores: list[dict]   ◄── reads: insights + critiques + profile_data
  │                       uncertainty_output: str
  │
  ├──► (parallel) VisualizerAgent
  │
  ▼
ReporterAgent ◄── insights + critiques + confidence_scores + uncertainty_output
  │
END
```

**LangGraph edges:**
```python
graph.add_edge("analyst",     "critic")
graph.add_edge("critic",      "uncertainty")
graph.add_edge("analyst",     "visualizer")   # parallel branch
graph.add_edge("uncertainty", "reporter")
graph.add_edge("visualizer",  "reporter")
```

---

## Inputs

The Estimator reads three keys from LangGraph state:

| Key | Type | Source |
|---|---|---|
| `insights` | `list[dict]` | AnalystAgent |
| `critiques` | `list[dict]` | CriticAgent |
| `profile_data` | `dict` | DataProfiler |

**How insights are matched to critiques:**

```python
critique_map = {c["insight_title"]: c for c in critiques}

for insight in insights:
    critique = critique_map.get(insight["title"], {})  # empty dict if no match
    score = estimator.estimate(insight, critique, profile_data)
```

The link is the `title` field. The CriticAgent **must** copy the insight's exact `title`
into `insight_title` without reformulation (see design.md §5.1).

---

## Driver 1 — Data Quality (rule-based, 0–25 pts)

> *"Is the dataset large enough and clean enough to support claims?"*

**Inputs:** `profile_data` only — no LLM needed.

**Algorithm:**

| Check | Points |
|---|---|
| Row count ≥ 1000 | 10 |
| Row count 300–999 | 7 |
| Row count 100–299 | 4 |
| Row count < 100 | 1 |
| Missing values < 5% | +10 |
| Missing values 5–20% | +6 |
| Missing values 20–40% | +3 |
| Missing values > 40% | +0 |
| Duplicate rows < 1% | +5 |
| Duplicate rows 1–10% | +3 |
| Duplicate rows > 10% | +0 |

**Maximum: 25 points.**

**Example rationale string:** `"1500 rows, 0.8% missing, 3 duplicates"` → 22/25.

---

## Driver 2 — Specificity (rule-based, 0–25 pts)

> *"Is the insight grounded in concrete, verifiable details?"*

**Inputs:** `insight` text + `profile_data` (column names list).

**Algorithm:**

| Check | Points |
|---|---|
| Mentions ≥ 1 column name that exists in profile | +8 |
| Mentions ≥ 2 column names | +4 (cumulative, max +12) |
| Contains ≥ 1 concrete number or percentage (regex `\d+(\.\d+)?%?`) | +8 |
| Contains ≥ 2 numbers | +3 (cumulative, max +11) |
| Recommendation is actionable (≥ 8 words, not "investigate further") | +2 |

**Maximum: 25 points.**

**Example rationale string:** `"References 'region' column, cites 480 orders (32%)"` → 22/25.

---

## Driver 3 — Statistical Evidence (LLM-based, 0–25 pts)

> *"Does the observation describe an effect large and consistent enough to be meaningful?"*

**Inputs passed to LLM:** insight text + relevant profile statistics (column stats for
referenced columns only — compact, not the full profile dump).

**What the LLM judges:**
- Is the effect size large enough to be practically significant?
- Is there a formal statistical claim (correlation, trend) without supporting evidence?
- Is the observation an artifact of sampling or a genuine pattern?

**LLM output:** integer 0–25 + one-sentence rationale.

**Example rationale:** `"Clear percentage (32%), large enough sample, but no formal significance test"` → 20/25.

**Fallback (LLM unavailable):** default score = 12/25, rationale = "LLM unavailable — default score applied".

---

## Driver 4 — Critic Assessment (LLM-based, 0–25 pts)

> *"How severely does the Critic's analysis undermine the insight?"*

**Inputs passed to LLM:** insight text + the matching critique dict
(`weaknesses`, `alternatives`, `verdict`, `confidence`).

**What the LLM judges:**
- Are the weaknesses the Critic identified fundamental or superficial?
- Are the alternative hypotheses more plausible than the original?
- Does the Critic's verdict (`supported` / `partially_supported` / `weak`) hold up?

**LLM output:** integer 0–25 + one-sentence rationale.

**Example rationale:** `"Critic identifies population density as confounding — valid but insight still has partial support"` → 12/25.

**Fallback hierarchy when critique is missing:**

| Situation | Behavior |
|---|---|
| No critique for this insight | Default score: 12/25, rationale = "No critique available" |
| Critique present but LLM fails | Map `verdict` → score: `supported`=20, `partially_supported`=12, `weak`=5 |
| Both missing | Default score: 12/25 |

---

## Output Contract

⚠️ The Reporter depends on this exact schema.

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

---

## Confidence Levels

| Score | Level | Icon | Meaning |
|---|---|---|---|
| 80–100% | `high` | 🟢 | Act on this insight with confidence |
| 50–79% | `medium` | 🟡 | Useful but verify before major decisions |
| 0–49% | `low` | 🔴 | Treat as hypothesis, needs more investigation |

---

## LangGraph State Keys

After the Estimator node runs, these keys are written to state:

| Key | Type | Used by |
|---|---|---|
| `confidence_scores` | `list[dict]` | Reporter |
| `uncertainty_output` | `str` (markdown) | Reporter (optional narrative block) |

---

## `agents/uncertainty.py` — Module Structure

```
agents/uncertainty.py
│
├── UncertaintyEstimator (class)
│   ├── estimate_all(insights, critiques, profile_data) → dict
│   │     Returns: {"confidence_scores": list[dict], "uncertainty_output": str}
│   ├── estimate(insight, critique, profile_data) → dict
│   ├── _compute_data_quality(profile_data) → tuple[int, str]       # rule-based
│   ├── _compute_specificity(insight, profile_data) → tuple[int, str]  # rule-based
│   ├── _compute_llm_scores(insight, critique) → dict               # LLM-based
│   └── _determine_level(score: int) → str
│
├── UncertaintyFormatter (class)
│   └── to_markdown(scores: list[dict]) → str
│
├── uncertainty_node(state: dict) → dict    # LangGraph entry point
│
└── _extract_json(raw: str) → dict | None   # helper
```

The `estimate_all` method builds the `critique_map` internally and calls `estimate`
per insight. The node function calls `estimate_all` and merges the result into state.

**The Estimator never crashes.** Every code path ends with a score (fallbacks always
apply). `estimate` is wrapped in `try/except` at the node level.

---

## Prompt Keys (`config/prompts.yaml`)

```yaml
uncertainty:
  system: |
    You are a statistical evidence evaluator. ...

  human: |
    ## Insight
    {insight_json}

    ## Relevant Profile Statistics
    {profile_digest}

    ## Critic Analysis
    {critique_json}

    Score the insight on two dimensions (0–25 each):
    1. statistical_evidence
    2. critic_assessment

    Reply ONLY with valid JSON:
    {
      "statistical_evidence": {"score": <int>, "reason": "<string>"},
      "critic_assessment":    {"score": <int>, "reason": "<string>"}
    }
```

Both LLM-based drivers are scored in **one LLM call** per insight to avoid doubling
the cost. The LLM returns both scores in a single JSON response.

---

## Configuration (`evaluation/config.py` additions)

```python
# Uncertainty Estimator
EVAL_UNCERTAINTY_HIGH_THRESHOLD: int   = int(os.getenv("EVAL_UNCERTAINTY_HIGH_THRESHOLD", "80"))
EVAL_UNCERTAINTY_MEDIUM_THRESHOLD: int = int(os.getenv("EVAL_UNCERTAINTY_MEDIUM_THRESHOLD", "50"))
```

Row count and missing % breakpoints are code constants (not env vars) — they encode
statistical conventions that shouldn't change per deployment.

Added to `.env.example` with documentation.

---

## Session State

```python
st.session_state["confidence_scores"] = list[dict] | None
```

Auto-cleared when a new file is uploaded. The Estimator runs automatically as part of
the analysis pipeline — no separate button click.

---

## UI — Insights Tab

### Cards mode

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🟡 Medium confidence  76%                      [priority: high]      │
│ Sales concentrated in Ile-de-France                                  │
│ ──────────────────────────────────────────────────────────────────── │
│ Observation: 32% of orders come from this region (480/1500).        │
│ Hypothesis:  Population density and logistics proximity.            │
│ Recommendation: Normalize by regional population before deciding.   │
│                                                                      │
│ ▼ Confidence breakdown                                               │
│   Data quality          ████████████████████░░░░  22/25             │
│   Specificity           ████████████████████░░░░  22/25             │
│   Statistical evidence  ████████████████████░░░░  20/25             │
│   Critic assessment     ████████████░░░░░░░░░░░░  12/25             │
│                                                                      │
│ 📋 Critic: Population density not considered. Could reflect          │
│    marketing spend or warehouse location.                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Markdown mode

Confidence badge prepended to insight heading:
```
### 🟡 [76%] Sales concentrated in Ile-de-France
```

---

## Reporter Integration

The `uncertainty_output` string (generated by `UncertaintyFormatter.to_markdown`) is
passed directly to the reporter prompt as `{uncertainty_output}`. It contains a compact
table:

```markdown
## Confidence Scores

| Insight | Score | Level | Main concern |
|---|---|---|---|
| Sales concentrated in IDF | 76% | 🟡 Medium | Critic: confounding factor |
| High age variance | 84% | 🟢 High | — |
| Missing Q4 revenue | 41% | 🔴 Low | 38% missing in revenue column |
```

The reporter uses this to qualify claims: *"The sales concentration finding has medium
confidence — the Critic's point about population density warrants further normalization
before acting on it."*

---

## Testing Strategy

### Unit tests — `tests/test_critic_uncertainty.py`

All tests run without a live LLM. `UncertaintyEstimator` is tested with:
- Mock profile dicts (no DataProfiler needed)
- Mock critique dicts (no CriticAgent needed)
- Patched `_compute_llm_scores` for unit tests (returns fixed scores)

| Test | What it checks |
|---|---|
| `test_data_quality_large_clean` | 1500 rows, 0.8% missing → 22/25 |
| `test_data_quality_small_dirty` | 80 rows, 45% missing → ≤ 5/25 |
| `test_data_quality_medium` | 300 rows, 15% missing → mid-range |
| `test_specificity_column_and_number` | References col + number → high score |
| `test_specificity_vague` | No columns, no numbers → low score |
| `test_specificity_weak_recommendation` | "investigate further" → no bonus points |
| `test_llm_scores_parsed` | Valid JSON → `statistical_evidence` + `critic_assessment` extracted |
| `test_llm_scores_invalid_json` | Malformed response → fallback defaults returned |
| `test_fallback_no_critique` | Missing critique → critic_assessment = 12/25 |
| `test_fallback_verdict_supported` | `verdict="supported"`, LLM fails → 20/25 |
| `test_fallback_verdict_weak` | `verdict="weak"`, LLM fails → 5/25 |
| `test_determine_level_high` | score ≥ 80 → "high" |
| `test_determine_level_medium` | score 50–79 → "medium" |
| `test_determine_level_low` | score < 50 → "low" |
| `test_estimate_single_insight` | Full dict returned with all 4 drivers |
| `test_estimate_all_returns_one_per_insight` | `len(result) == len(insights)` |
| `test_estimate_all_empty` | Empty list → no crash, empty result |
| `test_confidence_score_sum` | Sum of driver scores equals `confidence_score` |
| `test_to_markdown_output` | Markdown contains all insight titles and scores |
| `test_uncertainty_node_writes_state` | Node writes `confidence_scores` + `uncertainty_output` |

### Integration test (marked `@pytest.mark.integration`)

| Test | What it checks |
|---|---|
| `test_estimate_live_llm` | Live Ollama call; LLM drivers return integers 0–25 |

---

## Implementation Order

1. `evaluation/config.py` — add `EVAL_UNCERTAINTY_HIGH_THRESHOLD`, `EVAL_UNCERTAINTY_MEDIUM_THRESHOLD`
2. `config/prompts.yaml` — add `uncertainty.system` and `uncertainty.human`
3. `agents/uncertainty.py` — `UncertaintyEstimator`, `UncertaintyFormatter`, `uncertainty_node`
4. `app/main.py` — confidence badges in Insights tab cards + markdown mode
5. `agents/reporter.py` — accept `uncertainty_output` from state, inject into reporter prompt
6. `graph/` — add `uncertainty_node` between `critic_node` and `reporter_node`
7. `tests/test_critic_uncertainty.py` — 20 unit tests + 1 integration test
