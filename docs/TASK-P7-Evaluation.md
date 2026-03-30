# P7 — Evaluation System: LLM-as-Judge

## Overview

The evaluation system adds a quality-control layer on top of the existing pipeline. After each agent produces its output, an **EvaluationAgent** (acting as an LLM judge) scores that output against a set of rubric criteria and returns structured critiques. A new Streamlit tab surfaces these scores and critiques to the user.

The evaluation system is **non-blocking**: it does not alter or gate the pipeline. It runs after the fact, giving the user transparency into how trustworthy each output is.

---

## What Gets Evaluated

Every artifact produced by an agent is a candidate for evaluation:

| Artifact | Producing Agent | Evaluated? |
|----------|----------------|-----------|
| Profiler markdown | `ProfilerAgent` | Yes |
| Analyst insights (list of dicts) | `AnalystAgent` | Yes |
| Reporter markdown | `ReporterAgent` | Yes |
| Visualizer chart specs | `VisualizerAgent` | Yes (when implemented) |
| Generated code | `TextToCodeAgent` | Yes (when implemented) |

Each artifact type has its own **rubric** — a set of criteria tailored to what makes that artifact good.

---

## Evaluation Criteria per Artifact

### Profiler Output (markdown)

| Criterion | Description |
|-----------|-------------|
| **Grounding** | Do all statistics mentioned match the actual `DataProfile` values? |
| **Completeness** | Are all required sections present (Overview, Quality, Columns, Stats, Takeaways)? |
| **Clarity** | Is the language accessible to a non-technical reader? |
| **Specificity** | Does it avoid vague statements? Does it reference actual column names and real numbers? |

### Analyst Insights (per insight and aggregate)

| Criterion | Description |
|-----------|-------------|
| **Factual correctness** | Is the observation grounded in the actual data profile (correct column names, plausible values)? |
| **Relevance** | Is the insight meaningful for the dataset domain, not a generic observation? |
| **Actionability** | Does the recommendation lead to a concrete next step? |
| **Priority calibration** | Is the assigned priority (high/medium/low) appropriate relative to other insights? |
| **Diversity** | Do the insights cover different aspects (trend, anomaly, correlation, distribution) rather than repeating the same theme? |

### Reporter Output (markdown report)

| Criterion | Description |
|-----------|-------------|
| **Coherence** | Does the narrative flow logically from summary → description → insights → recommendations? |
| **Faithfulness** | Does it accurately represent the upstream analyst insights without introducing new claims? |
| **Language quality** | Is it executive-ready — concise, professional, no technical jargon for non-technical sections? |
| **Completeness** | Are all 6 required sections present and substantive? |
| **Actionability** | Are the recommendations specific and implementable? |

### Visualizer Chart Specs (future)

| Criterion | Description |
|-----------|-------------|
| **Chart-type fit** | Is the chosen chart type appropriate for the data distribution/relationship it represents? |
| **Correctness** | Do the axis mappings reference actual columns from the dataset? |
| **Insight linkage** | Does each chart directly support one of the analyst insights? |

### Generated Code (future)

| Criterion | Description |
|-----------|-------------|
| **Executability** | Does the code run without errors in a sandboxed environment? |
| **Correctness** | Does the output match the intended analysis? |
| **Safety** | Are there no dangerous imports or I/O operations? |

---

## Data Model

### `EvaluationCriterion`

```python
@dataclass
class EvaluationCriterion:
    name: str           # e.g. "factual_correctness"
    label: str          # Human-readable: "Factual Correctness"
    score: float        # 0.0 – 1.0
    rationale: str      # One-sentence explanation of the score
```

### `EvaluationResult`

```python
@dataclass
class EvaluationResult:
    artifact_type: str            # "profiler" | "analyst" | "reporter" | "visualizer" | "code"
    overall_score: float          # Weighted average of criteria scores (0.0 – 1.0)
    grade: str                    # "excellent" | "good" | "fair" | "poor"
    criteria: list[EvaluationCriterion]
    critique: str                 # 2-4 sentence overall critique
    suggestions: list[str]        # Concrete improvement suggestions (1-3 items)
    judge_model: str              # Model that produced this evaluation
    timestamp: str                # ISO datetime
```

### `InsightScore`

```python
@dataclass
class InsightScore:
    title: str
    factual_correctness: float   # 0.0 – 1.0
    relevance: float
    actionability: float
    note: str                    # one-sentence observation from the judge
```

### `AnalystEvaluationResult`

Extends `EvaluationResult` with per-insight detail (returned by the single hybrid analyst call):

```python
@dataclass
class AnalystEvaluationResult(EvaluationResult):
    per_insight: list[InsightScore]
```

### `PipelineEvaluation`

```python
@dataclass
class PipelineEvaluation:
    profiler_eval: EvaluationResult | None
    analyst_eval:  AnalystEvaluationResult | None
    reporter_eval: EvaluationResult | None
    pipeline_score: float         # Weighted average across all evaluated artifacts
```

These dataclasses live in `evaluation/schemas.py`.

---

## Architecture

### Files to Create / Modify

```
evaluation/
├── __init__.py
├── schemas.py          # EvaluationCriterion, EvaluationResult, PipelineEvaluation
├── rubrics.py          # Per-artifact rubric definitions (criteria names, weights, descriptions)
├── llm_judge.py        # EvaluationAgent — LLM-as-judge orchestration
├── validators.py       # Deterministic validators (grounding checks against DataProfile)
└── code_eval.py        # Sandboxed code execution eval (future)

config/
└── prompts.yaml        # Add "evaluation" section with judge system/human prompts

app/
└── main.py             # Add "🔍 Evaluation" tab

tests/
├── test_evaluation_schemas.py
├── test_evaluation_validators.py
└── test_evaluation_judge.py     # Marked @pytest.mark.integration
```

---

## `evaluation/rubrics.py`

Rubrics are plain data — no LLM involved. They define what the judge should score and how to weight each criterion.

```python
RUBRICS: dict[str, list[dict]] = {
    "profiler": [
        {"name": "grounding",    "label": "Grounding",    "weight": 0.35},
        {"name": "completeness", "label": "Completeness", "weight": 0.25},
        {"name": "clarity",      "label": "Clarity",      "weight": 0.25},
        {"name": "specificity",  "label": "Specificity",  "weight": 0.15},
    ],
    "analyst": [
        {"name": "factual_correctness", "label": "Factual Correctness", "weight": 0.35},
        {"name": "relevance",           "label": "Relevance",           "weight": 0.20},
        {"name": "actionability",       "label": "Actionability",       "weight": 0.20},
        {"name": "priority_calibration","label": "Priority Calibration","weight": 0.15},
        {"name": "diversity",           "label": "Diversity",           "weight": 0.10},
    ],
    "reporter": [
        {"name": "coherence",     "label": "Coherence",      "weight": 0.25},
        {"name": "faithfulness",  "label": "Faithfulness",   "weight": 0.30},
        {"name": "language",      "label": "Language Quality","weight": 0.20},
        {"name": "completeness",  "label": "Completeness",   "weight": 0.15},
        {"name": "actionability", "label": "Actionability",  "weight": 0.10},
    ],
}
```

---

## `evaluation/validators.py`

Deterministic checks that run before the LLM judge, and feed into the judge's context. These are fast and do not require an LLM call.

```python
def check_column_references(text: str, profile: DataProfile) -> dict:
    """
    Checks whether column names mentioned in text actually exist in the dataset.
    Returns: {"valid": [...], "invalid": [...], "unmentioned": [...]}
    """

def check_statistic_accuracy(text: str, profile: DataProfile, tolerance: float = 0.05) -> dict:
    """
    Extracts numeric values from text (percentages, counts, means) and compares
    them to actual DataProfile stats within tolerance.
    Returns: {"accurate": [...], "inaccurate": [...]}
    """

def check_required_sections(text: str, artifact_type: str) -> dict:
    """
    Checks whether required markdown headings are present for the given artifact type.
    Returns: {"present": [...], "missing": [...]}
    """

def check_insight_fields(insights: list[dict]) -> dict:
    """
    Checks whether each insight has all required fields with non-empty values.
    Returns: {"valid_count": int, "issues": [...]}
    """
```

The validator results are serialized as a compact JSON string and injected into the judge's human prompt so the LLM does not need to re-derive facts it cannot reliably verify.

---

## Judge Model

**`qwen3:14b` is used as the judge** (the text model, not the coder model).

Rationale: evaluation is a text reasoning task — assessing clarity, coherence, relevance, and faithfulness. `qwen2.5-coder:14b` is optimised for code generation and syntax understanding; using it for prose quality assessment would be using the wrong tool. `qwen3:14b` is the stronger reasoning model for this purpose.

The judge model is configured independently from the pipeline via a dedicated env var so it can be changed without touching profiler/analyst/reporter settings:

```
EVAL_JUDGE_MODEL=qwen3:14b   # defaults to this; override freely
```

In `LLMClient`, a new method `get_judge_llm()` reads this variable and returns the configured model using the same Ollama/Gemini factory pattern already in place.

---

## Context Window Strategy — Section-Aware Truncation

Long artifacts (especially the reporter output) are **not** chunked and averaged. Instead, a **section-aware truncation** function extracts each `##` heading and takes up to 300 characters of its body, then reassembles the result. This guarantees the judge sees a representative excerpt of every section rather than only the beginning of the document.

```python
MAX_SECTION_CHARS = 300   # per section body
MAX_INSIGHTS_FIELD_CHARS = 150   # per field within each insight dict

def truncate_for_judge(artifact: str, artifact_type: str) -> str:
    """
    For markdown artifacts (profiler, reporter): extract each ## section,
    truncate its body to MAX_SECTION_CHARS, reassemble.
    For analyst insights (JSON): truncate each insight's text fields to
    MAX_INSIGHTS_FIELD_CHARS while keeping all insights intact.
    Returns a string safe to embed in the judge prompt.
    """
```

Why not chunk + average?
- Chunking multiplies LLM calls (N chunks × 3 artifact types = many calls for one evaluation run).
- Averaging scores across chunks hides section-level weaknesses.
- The deterministic validators already handle factual checks (column names, section presence, stat accuracy) — the judge's primary job is reasoning about quality, for which a dense representative excerpt is sufficient.

---

## Analyst Insights — Single-Call Hybrid Scoring

All insights are sent in **one LLM call** and the judge returns **both aggregate scores and per-insight scores** in a single JSON response. This gives maximum richness at the cost of one call (not N).

### Extended `EvaluationResult` for analyst

The analyst evaluation extends the base schema with a `per_insight` field:

```python
@dataclass
class InsightScore:
    title: str
    factual_correctness: float   # 0.0 – 1.0
    relevance: float
    actionability: float
    note: str                    # one-sentence observation from the judge

@dataclass
class AnalystEvaluationResult(EvaluationResult):
    per_insight: list[InsightScore]   # one entry per insight
```

### Judge response schema for analyst

```json
{
  "factual_correctness": {"score": 0.70, "rationale": "..."},
  "relevance":           {"score": 0.80, "rationale": "..."},
  "actionability":       {"score": 0.65, "rationale": "..."},
  "priority_calibration":{"score": 0.60, "rationale": "..."},
  "diversity":           {"score": 0.75, "rationale": "..."},
  "per_insight": [
    {"title": "High Ile-de-France concentration", "factual_correctness": 0.9, "relevance": 0.8, "actionability": 0.7, "note": "Well-grounded in the column data."},
    {"title": "Missing values in revenue", "factual_correctness": 0.5, "relevance": 0.6, "actionability": 0.4, "note": "Recommendation is vague."}
  ],
  "overall_critique": "...",
  "suggestions": ["...", "..."]
}
```

The UI renders per-insight scores as a collapsible table within the Analyst panel.

---

## `evaluation/llm_judge.py`

The central class. Uses the existing `LLMClient` — no new LLM infrastructure needed.

```python
class EvaluationAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client.get_judge_llm()   # qwen3:14b via EVAL_JUDGE_MODEL

    def evaluate_profiler(
        self,
        profiler_output: str,
        profile: DataProfile,
    ) -> EvaluationResult:
        artifact = truncate_for_judge(profiler_output, "profiler")
        validation_report = {
            "column_check": check_column_references(profiler_output, profile),
            "section_check": check_required_sections(profiler_output, "profiler"),
        }
        return self._run_judge("profiler", artifact, validation_report)

    def evaluate_analyst(
        self,
        insights: list[dict],
        profile: DataProfile,
    ) -> AnalystEvaluationResult:
        artifact = truncate_for_judge(json.dumps(insights, indent=2), "analyst")
        validation_report = {
            "field_check": check_insight_fields(insights),
            "column_check": check_column_references(str(insights), profile),
        }
        return self._run_judge("analyst", artifact, validation_report)

    def evaluate_reporter(
        self,
        reporter_output: str,
        analyst_output: str,
    ) -> EvaluationResult:
        artifact = truncate_for_judge(reporter_output, "reporter")
        context  = truncate_for_judge(analyst_output, "reporter")   # same budget for context
        validation_report = {
            "section_check": check_required_sections(reporter_output, "reporter"),
        }
        return self._run_judge("reporter", artifact, validation_report, context=context)

    def _run_judge(
        self,
        artifact_type: str,
        artifact: str,
        validation_report: dict,
        context: str = "",
    ) -> EvaluationResult:
        """
        Builds the judge prompt from the rubric and validation report,
        calls the LLM, parses the structured JSON response.
        """
        rubric = RUBRICS[artifact_type]
        system_prompt = build_judge_system_prompt(rubric)
        human_prompt  = build_judge_human_prompt(artifact, validation_report, context)
        raw = call_llm_with_messages(system_prompt, human_prompt, self.llm)
        return parse_judge_response(raw, artifact_type)
```

### Judge Prompt Design

The system prompt tells the LLM its role as a strict but fair evaluator. It receives the rubric criteria and is instructed to:

1. Return a JSON object with one key per criterion, each containing `score` (0.0–1.0) and `rationale` (one sentence).
2. Return an `overall_critique` field (2-4 sentences).
3. Return a `suggestions` field (list of 1-3 concrete improvements).
4. For analyst artifacts, also return `per_insight` array (one entry per insight).
5. **Never hallucinate** — rely only on what is in the artifact and the validation report.

The human prompt injects:
- The artifact text/JSON (section-aware truncated)
- The pre-computed validation report (column references, section presence, stat accuracy)
- Any relevant upstream context, also truncated (e.g. analyst output when judging the reporter)

### Response Parsing

Same pattern as `AnalystAgent.extract_json()` — strip markdown fences, parse JSON, validate fields, fall back gracefully. If parsing fails, return an `EvaluationResult` with `overall_score=0.0` and the raw LLM text as the critique.

---

## `config/prompts.yaml` — New Section

```yaml
evaluation:
  judge:
    system: |
      You are a rigorous evaluator of AI-generated data analysis outputs.
      You will be given an artifact and a pre-computed validation report.
      Your job is to score the artifact against the provided rubric criteria.
      Be strict, precise, and grounded. Only reference what is actually in the artifact.
      Do not invent facts. Penalise vague or generic statements.
      Return a JSON object exactly matching the schema provided.

    human: |
      ## Artifact Type: {artifact_type}

      ## Rubric Criteria (score each 0.0–1.0):
      {rubric_json}

      ## Pre-computed Validation Report:
      {validation_report_json}

      ## Upstream Context (if any):
      {context}

      ## Artifact to Evaluate:
      {artifact}

      ## Required Output (JSON only):
      {{
        "<criterion_name>": {{"score": 0.0, "rationale": "..."}},
        ...
        "overall_critique": "...",
        "suggestions": ["...", "..."]
      }}
```

---

## Streamlit UI — `🔍 Evaluation` Tab

A new 7th tab is added to `app/main.py`.

### Session State Additions

```python
st.session_state["pipeline_eval"] = PipelineEvaluation | None
```

**Auto-invalidation on file change**: when `last_file` changes (new upload detected), `pipeline_eval` is reset to `None` immediately, clearing all displayed scores. The user must click "Run Evaluation" again for the new file.

```python
if st.session_state.get("last_file") != current_file_name:
    st.session_state["last_file"]     = current_file_name
    st.session_state["pipeline_eval"] = None   # invalidate on new file
```

### Tab Layout

```
🔍 Evaluation
├── [Run Evaluation] button (triggers EvaluationAgent for all available artifacts)
├── Pipeline Score: ██████░░░░  0.72 / 1.0  (progress bar + grade badge)
│
├── Profiler Quality              Analyst Quality              Reporter Quality
│   Score: 0.81 ✅               Score: 0.64 ⚠️              Score: 0.78 ✅
│   ─────────────────────        ─────────────────────        ─────────────────────
│   Grounding      0.90          Factual Correct. 0.55        Coherence     0.80
│   Completeness   0.80          Relevance        0.70        Faithfulness  0.75
│   Clarity        0.75          Actionability    0.65        Language      0.85
│   Specificity    0.80          Priority Cal.    0.60        Completeness  0.70
│                                Diversity        0.80        Actionability 0.65
│
│   Critique: "The profiler..."  Critique: "Two insights..."  Critique: "The report..."
│   Suggestions:                 Suggestions:                  Suggestions:
│   • ...                        • ...                         • ...
│
│                               ▼ Per-insight scores (expander)
│                               ┌──────────────────────────────────────────────────┐
│                               │ Insight                  Correct  Relev  Action  │
│                               │ High Ile-de-France conc.  0.90    0.80   0.70   │
│                               │ Missing values in revenue 0.50    0.60   0.40   │
│                               │ ...                                              │
│                               └──────────────────────────────────────────────────┘
```

### Score Colours

| Score | Colour | Grade |
|-------|--------|-------|
| 0.85 – 1.0 | Green | Excellent |
| 0.70 – 0.84 | Blue | Good |
| 0.50 – 0.69 | Orange | Fair |
| 0.0 – 0.49 | Red | Poor |

### Components Used

- `st.progress(score)` — score bars per criterion
- `st.metric(label, value, delta)` — overall score with delta vs. threshold
- `st.expander` — collapse/expand per-criterion details and per-insight table
- `st.columns(3)` — side-by-side agent evaluation panels
- `st.dataframe` — per-insight scores table (colour-mapped with `pandas.Styler`)
- `st.warning` / `st.success` / `st.error` — grade-based alert boxes

---

## Integration with the Existing Pipeline

No existing agent code changes. The evaluation is triggered explicitly by the user via the "Run Evaluation" button. It reads from session state:

```python
# Already available in session state after pipeline runs:
profiler_output  = st.session_state["ai_description"]
analyst_insights = st.session_state["analyst_insights"]
analyst_markdown = st.session_state["analyst_markdown"]
reporter_output  = st.session_state["reporter_output"]
profile          = st.session_state["profile"]          # DataProfile object — already there

# New:
pipeline_eval    = st.session_state["pipeline_eval"]    # PipelineEvaluation
```

The `EvaluationAgent` is instantiated fresh each time (same as other agents), using the same `LLMClient` singleton.

---

## Testing Strategy

All unit tests run without an LLM (mocked responses). Integration tests are marked `@pytest.mark.integration` and require a running Ollama instance.

### `tests/test_evaluation_schemas.py`

| Test | What it checks |
|------|---------------|
| `test_evaluation_criterion_construction` | `EvaluationCriterion` stores name, label, score, rationale correctly |
| `test_evaluation_result_grade_excellent` | score ≥ 0.85 → grade == "excellent" |
| `test_evaluation_result_grade_good` | 0.70 ≤ score < 0.85 → grade == "good" |
| `test_evaluation_result_grade_fair` | 0.50 ≤ score < 0.70 → grade == "fair" |
| `test_evaluation_result_grade_poor` | score < 0.50 → grade == "poor" |
| `test_pipeline_evaluation_score_average` | `pipeline_score` == weighted average of non-None artifact scores |
| `test_pipeline_evaluation_partial` | `pipeline_score` ignores `None` artifacts |
| `test_analyst_evaluation_result_per_insight` | `AnalystEvaluationResult` stores `per_insight` list correctly |
| `test_insight_score_fields` | `InsightScore` stores all five fields |

### `tests/test_evaluation_rubrics.py`

| Test | What it checks |
|------|---------------|
| `test_profiler_rubric_weights_sum_to_one` | profiler weights sum to 1.0 |
| `test_analyst_rubric_weights_sum_to_one` | analyst weights sum to 1.0 |
| `test_reporter_rubric_weights_sum_to_one` | reporter weights sum to 1.0 |
| `test_all_rubric_entries_have_required_keys` | every entry has `name`, `label`, `weight` |
| `test_weight_is_float_between_zero_and_one` | no weight < 0 or > 1 |

### `tests/test_evaluation_validators.py`

| Test | What it checks |
|------|---------------|
| `test_check_column_references_valid` | known column name → appears in `valid` list |
| `test_check_column_references_invalid` | fabricated column name → appears in `invalid` list |
| `test_check_column_references_empty_text` | empty string → all columns in `unmentioned` |
| `test_check_statistic_accuracy_within_tolerance` | number within 5% of real stat → `accurate` |
| `test_check_statistic_accuracy_outside_tolerance` | number > 5% off → `inaccurate` |
| `test_check_statistic_accuracy_no_numbers` | text with no numbers → empty results, no crash |
| `test_check_required_sections_all_present` | markdown with all headings → `missing` is empty |
| `test_check_required_sections_some_missing` | markdown missing one heading → correct heading in `missing` |
| `test_check_insight_fields_valid` | well-formed insight list → `issues` is empty |
| `test_check_insight_fields_missing_key` | insight missing `recommendation` → flagged in `issues` |
| `test_check_insight_fields_empty_value` | insight with `title: ""` → flagged in `issues` |

### `tests/test_evaluation_judge.py`

| Test | Type | What it checks |
|------|------|---------------|
| `test_truncate_for_judge_profiler` | unit | output contains all section headings, each body ≤ 300 chars |
| `test_truncate_for_judge_analyst` | unit | each insight's text fields ≤ 150 chars, all insights present |
| `test_truncate_for_judge_no_crash_on_empty` | unit | empty string → returns empty string, no exception |
| `test_parse_judge_response_valid_json` | unit | known valid JSON → `EvaluationResult` with correct scores |
| `test_parse_judge_response_with_markdown_fences` | unit | JSON wrapped in ```json``` → parsed correctly |
| `test_parse_judge_response_invalid_json` | unit | malformed JSON → fallback `EvaluationResult` with score 0.0 |
| `test_parse_judge_response_analyst_per_insight` | unit | analyst JSON with `per_insight` → `AnalystEvaluationResult` |
| `test_build_judge_system_prompt_contains_criteria` | unit | output string contains each criterion name from rubric |
| `test_build_judge_human_prompt_contains_artifact` | unit | artifact text appears in output string |
| `test_evaluate_profiler_good_artifact` | integration | good profiler output → `overall_score` ≥ 0.60 |
| `test_evaluate_profiler_bad_artifact` | integration | lorem ipsum → `overall_score` < 0.50 |
| `test_evaluate_analyst_score_order` | integration | good insights score higher than empty list |
| `test_evaluate_reporter_good_artifact` | integration | full reporter output → `overall_score` ≥ 0.60 |

### Mock Strategy

Unit tests mock the LLM call at `utils.llm.call_llm_with_messages` to return a known valid JSON string — same pattern as `test_analyst.py`. The `DataProfile` fixture from `conftest.py` is reused for validator tests.

---

## LangFuse Tracing (Optional Enhancement)

Each `EvaluationAgent` call can be wrapped with a LangFuse span:

```python
trace = langfuse.trace(name="evaluation", input={"artifact_type": artifact_type})
span = trace.span(name="llm_judge")
# ... call LLM ...
span.end(output=result)
```

This makes evaluation quality visible in the existing LangFuse dashboard alongside profiler/analyst/reporter traces.

---

## Implementation Order

1. `evaluation/schemas.py` — `EvaluationCriterion`, `EvaluationResult`, `InsightScore`, `AnalystEvaluationResult`, `PipelineEvaluation`
2. `evaluation/rubrics.py` — rubric definitions with weights
3. `evaluation/validators.py` — deterministic checks (depends on `DataProfile`)
4. `utils/llm.py` — add `get_judge_llm()` method reading `EVAL_JUDGE_MODEL`
5. `config/prompts.yaml` — add evaluation prompts section
6. `evaluation/llm_judge.py` — truncation helpers + `EvaluationAgent` (depends on 1–5)
7. `app/main.py` — add Evaluation tab + auto-invalidation logic
8. Tests (can be written in parallel with steps 1–6)

---

## Design Decisions Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| Context window | Section-aware truncation (300 chars/section body) | Preserves coverage of all sections; avoids multiplying LLM calls from chunking; deterministic validators handle factual checks so the judge only needs representative text |
| Judge model | `qwen3:14b` via `EVAL_JUDGE_MODEL` env var | Evaluation is a text reasoning task; `qwen2.5-coder:14b` is optimised for code generation and not suited for prose quality assessment |
| Re-evaluation on refresh | Auto-invalidate `pipeline_eval` when `last_file` changes | Evaluation is tied to a specific file; stale scores from a previous dataset would mislead the user |
| Insight scoring granularity | Single LLM call → aggregate + per-insight in one response | One call gives both aggregate criteria scores and per-insight breakdown; avoids N×LLM calls while preserving richness |
