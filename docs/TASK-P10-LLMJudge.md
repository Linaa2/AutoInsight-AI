# P10 — LLM-as-Judge Evaluation Agent

## Core Idea

The LLM-as-Judge asks: *"How good is the quality of each pipeline output?"*

It is the only evaluation component that looks at **output quality as a whole** rather
than the reasoning behind a specific insight. It answers questions like:
- Is the profiler report grounded in actual data and clearly structured?
- Are the analyst insights diverse, relevant, and actionable?
- Does the reporter faithfully represent everything that was found?

| | Critic Agent | Uncertainty Estimator | **LLM-as-Judge** |
|---|---|---|---|
| **Question** | Is the reasoning sound? | How reliable is the data? | How good is the output quality? |
| **Scope** | Per insight | Per insight | Per artifact (profiler / analyst / reporter) |
| **Pipeline position** | In-pipeline (after Analyst) | In-pipeline (after Critic) | Post-run (user-triggered) |
| **Method** | LLM | Rules + LLM | LLM + deterministic validators |
| **Trigger** | Automatic | Automatic | Manual ("🔍 Evaluate" button) |
| **Cost** | 1 LLM call per insight | ~1 LLM call per insight | 3 LLM calls (one per artifact) |

---

## Position in the Workflow

```
PipelineState (after full pipeline run)
    │
    │  profile_markdown   ──────────────────────────────────────────┐
    │  insights           ────────────────────────────────────────┐ │
    │  insights_markdown  ──────────────────────────────────────┐ │ │
    │  report_markdown    ────────────────────────────────────┐ │ │ │
    │  critiques          ──────────────────────────────────┐ │ │ │ │
    │  confidence_scores  ────────────────────────────────┐ │ │ │ │ │
    │                                                      │ │ │ │ │ │
    └──────────────────────────────────────────────────────▼─▼─▼─▼─▼─▼
                                                    EvaluationAgent
                                                    (user-triggered)
                                                           │
                                          ┌────────────────┼────────────────┐
                                          ▼                ▼                ▼
                                  profiler_eval     analyst_eval     reporter_eval
                                  EvaluationResult  AnalystEval      EvaluationResult
                                                    Result
                                          └────────────────┼────────────────┘
                                                           ▼
                                                  PipelineEvaluation
                                                  (stored in session state)
                                                           │
                                                           ▼
                                                  🔍 Evaluation Tab (UI)
```

The judge **reads** from `PipelineState` but **writes nothing back** to it.
Results live exclusively in `st.session_state["pipeline_eval"]`.

---

## What Gets Evaluated and How

### Artifact 1 — Profiler Output

**Input:** `profile_markdown` + deterministic validator results
**Question:** *Is the profiler report accurate, complete, and readable?*

| Criterion | Weight | What it checks |
|---|---|---|
| `grounding` | 0.35 | Column names and stats are correctly cited; no hallucinated values |
| `completeness` | 0.25 | All required sections present; all column types covered |
| `clarity` | 0.25 | Writing is clear, well-structured, avoids jargon |
| `specificity` | 0.15 | Findings reference concrete numbers, not vague generalisations |

Required sections (auto-checked before LLM call):
`Dataset Overview`, `Data Quality Assessment`, `Column-by-Column Analysis`,
`Statistical Highlights`, `Key Takeaways & Recommendations`

---

### Artifact 2 — Analyst Insights

**Input:** `insights` list + `profile_markdown` + optional `critiques` + optional `confidence_scores`
**Question:** *Are the insights diverse, correct, and worth acting on?*

| Criterion | Weight | What it checks |
|---|---|---|
| `factual_correctness` | 0.35 | Claims are consistent with profile statistics |
| `relevance` | 0.20 | Insights address meaningful business questions, not trivia |
| `actionability` | 0.20 | Recommendations are concrete and implementable |
| `priority_calibration` | 0.15 | High-priority flags are proportional to actual business impact |
| `diversity` | 0.10 | Insights cover different aspects; no near-duplicate observations |

**Hybrid scoring**: The LLM scores all aggregate rubric criteria **and** provides a
per-insight breakdown (`factual_correctness`, `relevance`, `actionability`) in a
single LLM call — no N×LLM calls.

**Optional context injection**: If `critiques` are present in state (from CriticAgent),
a compact summary is injected into the judge prompt so the LLM knows which insights
were already flagged for weak reasoning. If `confidence_scores` are present, low-
confidence insights are highlighted. This avoids re-discovering what is already known.

---

### Artifact 3 — Reporter Output

**Input:** `report_markdown` + `insights_markdown` + `profile_markdown`
**Question:** *Does the report faithfully and coherently tell the story?*

| Criterion | Weight | What it checks |
|---|---|---|
| `faithfulness` | 0.30 | All key findings from profiler and analyst are represented |
| `coherence` | 0.25 | Sections flow logically; the narrative is consistent |
| `language` | 0.20 | Accessible to a non-technical reader; no raw jargon |
| `completeness` | 0.15 | All required sections present and substantive |
| `actionability` | 0.10 | Recommendations are concrete and prioritised |

Required sections: `Executive Summary`, `Dataset Description`, `Key Insights`,
`Visualizations`, `Recommendations`, `Limitations & Next Steps`

---

## Deterministic Validators (run before every LLM call)

Four validators inject ground truth into every judge prompt so the LLM is not
asked to re-derive facts it cannot reliably verify:

| Validator | Input | Output injected into prompt |
|---|---|---|
| `check_required_sections(text, artifact_type)` | Markdown text | Missing section names |
| `check_column_references(text, profile_data)` | Markdown + profile | Present / unmentioned columns |
| `check_statistic_accuracy(text, profile_data)` | Markdown + profile | Accurate / inaccurate stats |
| `check_insight_fields(insights)` | Insights list | Valid count / field issues |

Running validators first means the LLM prompt already says:
*"The following sections are missing: …"* or *"The following stat appears wrong: …"*
The LLM confirms or nuances, rather than discovering from scratch.

---

## Data Model

### `EvaluationCriterion`
```python
@dataclass
class EvaluationCriterion:
    name: str       # e.g. "grounding"
    label: str      # "Grounding"
    score: float    # 0.0 – 1.0
    rationale: str  # one sentence explaining the score
    weight: float   # contribution to overall score
```

### `EvaluationResult`
```python
@dataclass
class EvaluationResult:
    artifact_type: str           # "profiler" | "analyst" | "reporter"
    overall_score: float         # weighted composite 0.0 – 1.0
    grade: str                   # "excellent" | "good" | "fair" | "poor"
    criteria: list[EvaluationCriterion]
    critique: str                # 2–3 sentence overall assessment
    suggestions: list[str]       # 2–4 concrete improvement suggestions
    judge_model: str             # model that produced this result
    timestamp: str               # ISO-8601

    @staticmethod
    def compute_grade(score: float) -> str:
        # Thresholds from evaluation.config (env-var configurable)
        ...
```

### `InsightScore`
```python
@dataclass
class InsightScore:
    title: str
    factual_correctness: float   # 0.0 – 1.0
    relevance: float             # 0.0 – 1.0
    actionability: float         # 0.0 – 1.0
    note: str                    # one-sentence per-insight note from the judge
```

### `AnalystEvaluationResult` (extends `EvaluationResult`)
```python
@dataclass
class AnalystEvaluationResult(EvaluationResult):
    per_insight: list[InsightScore] = field(default_factory=list)
```

### `PipelineEvaluation`
```python
@dataclass
class PipelineEvaluation:
    profiler_eval:  EvaluationResult | None = None
    analyst_eval:   AnalystEvaluationResult | None = None
    reporter_eval:  EvaluationResult | None = None

    @property
    def pipeline_score(self) -> float:
        # Average of available artifact scores
        ...

    @property
    def pipeline_grade(self) -> str:
        return EvaluationResult.compute_grade(self.pipeline_score)
```

---

## Module Structure

```
evaluation/
├── config.py            # All env-var-backed thresholds (already exists — extend it)
├── schemas.py           # EvaluationCriterion, EvaluationResult, InsightScore,
│                        # AnalystEvaluationResult, PipelineEvaluation
├── rubrics.py           # RUBRICS dict, REQUIRED_SECTIONS dict
├── validators.py        # 4 deterministic validator functions
└── llm_judge.py         # EvaluationAgent class + prompt builders + parsers
```

### `evaluation/llm_judge.py` internals

```
evaluation/llm_judge.py
│
├── truncate_for_judge(text, max_chars_per_section) -> str
│       Section-aware truncation: extract each ## heading + cap body to
│       EVAL_MAX_SECTION_CHARS chars. Guarantees all sections are seen.
│
├── build_judge_system_prompt(artifact_type) -> str
│
├── build_judge_human_prompt(artifact_type, content, validator_results,
│                            context_hints) -> str
│       context_hints: optional dict with critic/uncertainty summaries
│
├── parse_judge_response(raw_json, artifact_type) -> EvaluationResult
│
├── _fallback_result(artifact_type, reason) -> EvaluationResult
│
└── class EvaluationAgent:
        .evaluate_profiler(profile_markdown, profile_data) -> EvaluationResult
        .evaluate_analyst(insights, profile_markdown, profile_data,
                          critiques=None, confidence_scores=None)
                         -> AnalystEvaluationResult
        .evaluate_reporter(report_markdown, profile_markdown,
                           insights_markdown) -> EvaluationResult
```

---

## Configuration (`evaluation/config.py` additions)

```python
# LLM Judge
# Defaults to the active provider's text model when unset.
EVAL_JUDGE_MODEL: str = ...
EVAL_JUDGE_TIMEOUT: int = int(os.getenv("EVAL_JUDGE_TIMEOUT", "300"))
EVAL_MAX_SECTION_CHARS: int  = int(os.getenv("EVAL_MAX_SECTION_CHARS",  "300"))
EVAL_MAX_INSIGHT_FIELD_CHARS: int = int(os.getenv("EVAL_MAX_INSIGHT_FIELD_CHARS", "150"))

# Grade thresholds
EVAL_EXCELLENT_THRESHOLD: float = float(os.getenv("EVAL_EXCELLENT_THRESHOLD", "0.85"))
EVAL_GOOD_THRESHOLD: float      = float(os.getenv("EVAL_GOOD_THRESHOLD",      "0.70"))
EVAL_FAIR_THRESHOLD: float      = float(os.getenv("EVAL_FAIR_THRESHOLD",      "0.50"))
```

---

## LLM Architecture

### Single call per artifact, hybrid for analyst

| Artifact | LLM calls | What the call returns |
|---|---|---|
| Profiler | 1 | Rubric scores + overall critique + suggestions |
| Analyst | 1 | Rubric scores + per-insight breakdown in one JSON |
| Reporter | 1 | Rubric scores + overall critique + suggestions |

**Total: 3 LLM calls** when all three artifacts are evaluated.

Judge calls use `EVAL_JUDGE_MODEL` and `EVAL_JUDGE_TIMEOUT`, so evaluation can
run with a lighter model and/or a longer timeout budget than the main pipeline.

### Section-aware truncation

Long markdown artifacts are truncated per section to `EVAL_MAX_SECTION_CHARS` before
being sent to the judge. This ensures all sections are visible to the LLM without
token overflow, and avoids the chunking + averaging approach (which hides per-section
weaknesses).

### Prompt structure (all three artifacts)

```
SYSTEM: judge role + rubric definitions + scoring instructions + JSON schema
HUMAN:
  [Optional] Prior analysis signals:
    - Critic flags: "Insight X was flagged for causal overclaim"
    - Uncertainty: "Insight Y has low confidence (42%)"

  [Validators output]:
    - Missing sections: [...]
    - Column references: present=[...], unmentioned=[...]
    - Stat accuracy: accurate=[...], inaccurate=[...]

  [Artifact text (truncated)]

  Return ONLY valid JSON matching the schema above.
```

### JSON output schema

**Profiler / Reporter:**
```json
{
  "criteria": {
    "grounding":    {"score": 0.82, "rationale": "..."},
    "completeness": {"score": 0.90, "rationale": "..."},
    "clarity":      {"score": 0.75, "rationale": "..."},
    "specificity":  {"score": 0.68, "rationale": "..."}
  },
  "critique":    "Overall assessment in 2-3 sentences.",
  "suggestions": ["...", "..."]
}
```

**Analyst (single call, aggregate + per-insight):**
```json
{
  "criteria": {
    "factual_correctness":  {"score": 0.80, "rationale": "..."},
    "relevance":            {"score": 0.85, "rationale": "..."},
    "actionability":        {"score": 0.75, "rationale": "..."},
    "priority_calibration": {"score": 0.70, "rationale": "..."},
    "diversity":            {"score": 0.90, "rationale": "..."}
  },
  "critique":    "...",
  "suggestions": ["..."],
  "per_insight": [
    {
      "title": "...",
      "factual_correctness": 0.85,
      "relevance": 0.90,
      "actionability": 0.80,
      "note": "..."
    }
  ]
}
```

---

## Context Injection from Critic and Uncertainty

When results from earlier pipeline stages are available in state, they are injected
as compact context blocks — they do NOT replace the judge's own scoring.

**From Critic (if `critiques` available):**
```
## Prior Critic Flags
- "Sales concentrated in IDF" — verdict: partially_supported. Weakness: population density not considered.
- "Revenue outliers" — verdict: weak. Weakness: causal overclaim.
```

**From Uncertainty (if `confidence_scores` available):**
```
## Data Confidence Signals
- "Sales concentrated in IDF" — medium confidence (65%). Small sample.
- "Revenue outliers" — low confidence (41%). 38% missing in revenue.
```

This context appears in the analyst judge prompt only. The judge is explicitly told:
*"Use this as context, not as a substitute for your own assessment."*

---

## Session State

```python
st.session_state["pipeline_eval"] = PipelineEvaluation | None
```

- Cleared when a new file is uploaded
- Cleared when the pipeline is re-run (new analysis invalidates prior evaluation)
- Persists across tab switches within the same analysis session

---

## UI — `🔍 Evaluation` Tab

### Layout

```
🔍 Evaluation

[▶ Run Evaluation]   (triggers all 3 LLM calls with a spinner)

──────────────────────────────────────────────────────
Pipeline Score: 0.78   Grade: 🟡 Good
──────────────────────────────────────────────────────

[📊 Profiler]   [💡 Analyst]   [📄 Reporter]   (sub-tabs)
```

### Per-artifact panel

```
📊 Profiler Evaluation  ·  Score: 0.84  ·  Grade: 🟢 Excellent

  Grounding        ████████░░  0.82
  Completeness     █████████░  0.90
  Clarity          ████████░░  0.75
  Specificity      ███████░░░  0.68

  💬 "The profiler correctly cites the age column's 20% missingness and provides
  accurate mean/std statistics. However, the temporal analysis section is brief."

  ✅ Suggestions:
  • Add a correlation hint between age and salary in Statistical Highlights.
  • Expand the datetime column analysis with trend direction.
```

### Analyst panel — per-insight table

When `AnalystEvaluationResult.per_insight` is populated, a `st.dataframe` inside an
expander shows all insights with their individual scores side-by-side.

### Grade thresholds and icons

| Score | Grade | Icon |
|---|---|---|
| ≥ 0.85 | `excellent` | 🟢 |
| ≥ 0.70 | `good` | 🟡 |
| ≥ 0.50 | `fair` | 🟠 |
| < 0.50 | `poor` | 🔴 |

---

## `evaluation/rubrics.py`

```python
RUBRICS: dict[str, list[dict]] = {
    "profiler": [
        {"name": "grounding",     "label": "Grounding",     "weight": 0.35},
        {"name": "completeness",  "label": "Completeness",  "weight": 0.25},
        {"name": "clarity",       "label": "Clarity",       "weight": 0.25},
        {"name": "specificity",   "label": "Specificity",   "weight": 0.15},
    ],
    "analyst": [
        {"name": "factual_correctness",  "label": "Factual Correctness",  "weight": 0.35},
        {"name": "relevance",            "label": "Relevance",            "weight": 0.20},
        {"name": "actionability",        "label": "Actionability",        "weight": 0.20},
        {"name": "priority_calibration", "label": "Priority Calibration", "weight": 0.15},
        {"name": "diversity",            "label": "Diversity",            "weight": 0.10},
    ],
    "reporter": [
        {"name": "faithfulness",  "label": "Faithfulness",     "weight": 0.30},
        {"name": "coherence",     "label": "Coherence",        "weight": 0.25},
        {"name": "language",      "label": "Language Quality", "weight": 0.20},
        {"name": "completeness",  "label": "Completeness",     "weight": 0.15},
        {"name": "actionability", "label": "Actionability",    "weight": 0.10},
    ],
}

REQUIRED_SECTIONS: dict[str, list[str]] = {
    "profiler": [
        "Dataset Overview", "Data Quality Assessment",
        "Column-by-Column Analysis", "Statistical Highlights",
        "Key Takeaways",
    ],
    "reporter": [
        "Executive Summary", "Dataset Description", "Key Insights",
        "Visualizations", "Recommendations", "Limitations",
    ],
}
```

Weights within each artifact sum to 1.0. Verified by tests.

---

## Testing Strategy

### Unit tests — `tests/test_llm_judge.py`

All unit tests run without a live LLM. `EvaluationAgent` methods are tested via
patched `call_llm_with_messages`.

| Test | What it checks |
|---|---|
| `test_truncate_preserves_all_sections` | All `##` headings survive truncation |
| `test_truncate_caps_body_per_section` | Each section body is ≤ `EVAL_MAX_SECTION_CHARS` |
| `test_truncate_short_text_unchanged` | Text below limit is returned as-is |
| `test_required_sections_all_present` | Validator returns no missing sections |
| `test_required_sections_missing_one` | Missing section is flagged correctly |
| `test_column_references_present` | Referenced columns correctly identified |
| `test_column_references_unmentioned` | Unmentioned columns listed |
| `test_stat_accuracy_correct` | Correctly cited stat → accurate list |
| `test_stat_accuracy_wrong` | Wrong stat → inaccurate list |
| `test_insight_fields_valid` | All fields present → valid_count = N |
| `test_insight_fields_missing_title` | Missing title → issue reported |
| `test_parse_profiler_response_valid` | Valid JSON → `EvaluationResult` |
| `test_parse_analyst_response_with_per_insight` | `per_insight` list parsed |
| `test_parse_invalid_json_returns_fallback` | Garbage → `_fallback_result` used |
| `test_parse_partial_criteria_fill_defaults` | Missing criteria filled with 0.5 |
| `test_overall_score_is_weighted_average` | Correct weighted composite |
| `test_compute_grade_excellent` | score ≥ 0.85 → "excellent" |
| `test_compute_grade_poor` | score < 0.50 → "poor" |
| `test_evaluate_profiler_calls_llm_once` | Exactly 1 LLM call |
| `test_evaluate_analyst_calls_llm_once` | Exactly 1 LLM call (not N) |
| `test_evaluate_reporter_calls_llm_once` | Exactly 1 LLM call |
| `test_evaluate_analyst_with_critic_context` | Critic summary injected in prompt |
| `test_evaluate_analyst_with_uncertainty_context` | Uncertainty injected in prompt |
| `test_pipeline_evaluation_score_average` | `pipeline_score` = mean of available |
| `test_pipeline_evaluation_grade_derived` | `pipeline_grade` from `pipeline_score` |
| `test_evaluate_profiler_no_llm_fallback` | LLM failure → fallback result, no crash |

### Integration tests (marked `@pytest.mark.integration`)

| Test | What it checks |
|---|---|
| `test_evaluate_profiler_live` | Live LLM; all criteria present; score ∈ [0,1] |
| `test_evaluate_analyst_live` | Live LLM; per_insight populated |
| `test_evaluate_reporter_live` | Live LLM; suggestions non-empty |

---

## Implementation Order

1. `evaluation/config.py` — add `EVAL_JUDGE_MODEL`, `EVAL_MAX_SECTION_CHARS`,
   `EVAL_MAX_INSIGHT_FIELD_CHARS`, `EVAL_EXCELLENT_THRESHOLD`, `EVAL_GOOD_THRESHOLD`,
   `EVAL_FAIR_THRESHOLD`
2. `evaluation/schemas.py` — `EvaluationCriterion`, `EvaluationResult`,
   `InsightScore`, `AnalystEvaluationResult`, `PipelineEvaluation`
3. `evaluation/rubrics.py` — `RUBRICS`, `REQUIRED_SECTIONS`
4. `evaluation/validators.py` — 4 validator functions
5. `evaluation/llm_judge.py` — `EvaluationAgent` + helpers
6. `config/prompts.yaml` — add `evaluation.judge` and `evaluation.analyst_judge` sections
7. `app/main.py` — `🔍 Evaluation` tab, `_render_evaluation()` + `_render_artifact_panel()`
8. `.env.example` — 6 new vars with documentation
9. `tests/test_llm_judge.py` — 26 unit + 3 integration tests

---

## Open Questions Before Implementation

1. **Scope of context injection**: Should Critic and Uncertainty context be injected
   into the judge prompts automatically, or only when the user explicitly enables
   "deep evaluation" mode? Auto-injection gives richer results but longer prompts.

2. **Selective evaluation**: Should the user be able to evaluate only one artifact
   (e.g. "just judge the reporter") rather than all three at once? Running all three
   costs 3 LLM calls — useful to split for faster feedback.

3. **Re-evaluation guard**: If the user re-runs the full pipeline (new analysis), the
   prior evaluation should be cleared. But if the user only changes the LLM provider
   setting, should old evaluation results be preserved or cleared?

4. **Grade display in Report tab**: Should the final report include the judge's
   quality grades as a metadata block (e.g. "Report quality: 🟢 Excellent"), or keep
   the Report tab clean for non-technical stakeholders?

5. **`per_insight` table scope**: Should the per-insight table in the Analyst panel
   also display the Critic's `verdict` and Uncertainty's `confidence_level` alongside
   the judge's `factual_correctness`? This would give a unified 3-signal view per insight.
