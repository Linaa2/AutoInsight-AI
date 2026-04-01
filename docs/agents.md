# Agents

This document summarizes the behavior and contracts of the main agents.

## Agent Overview

| Agent | Type | Model tier | Primary responsibility |
|---|---|---|---|
| `ProfilerAgent` | LLM agent | `light` | Turn deterministic profile context into readable markdown |
| `AnalystAgent` | LLM agent | `text` | Generate structured business insights |
| `CriticAgent` | LLM agent | `text` | Challenge insights and identify weaknesses |
| `UncertaintyEstimator` | Hybrid | `light` plus rules | Score insight confidence |
| `VisualizerAgent` | LLM agent | `code` | Propose chart specs and code |
| `ReporterAgent` | LLM agent | `text` | Write the executive report |
| `RAGAgent` | Retrieval service | none | Retrieve and store reusable context |
| `EvaluationAgent` | LLM agent | configurable | Judge output quality after the run |

## ProfilerAgent

### Role

Transforms a deterministic `DataProfile` into a readable markdown description.

### Inputs

- `DataProfile` from `tools/profiler_engine.py`
- callbacks for tracing
- optional detail mode

### Outputs

- markdown profile report

### Notable implementation details

- uses `build_profiler_prompt_context()` to send a compact summary instead of the full raw profile
- uses the `profiler` prompt section from `config/prompts.yaml`
- routed through the `light` model tier for latency

## AnalystAgent

### Role

Generates 3 to 6 structured insights from the profile and a small sample of the data.

### Inputs

- profiler markdown
- profile data
- textual sample of the dataset

### Outputs

- `insights`: list of structured insight dicts
- `insights_markdown`

### Insight schema

Each insight includes:

- `title`
- `observation`
- `hypothesis`
- `recommendation`
- `priority`
- optional category after categorization

### Notable implementation details

- supports keyword or LLM-based categorization
- validates and normalizes generated insights
- falls back gracefully when strict JSON output is not respected

## CriticAgent

### Role

Performs an adversarial review of each insight produced by the analyst.

### Inputs

- analyst insights
- profile digest or profile data

### Outputs

- `critiques`
- `critic_output`

### Critique schema

Each critique includes:

- `insight_title`
- `strengths`
- `weaknesses`
- `alternatives`
- `confidence`
- `verdict`

### Notable implementation details

- critiques are batched for lower latency
- each insight still gets a critique, even on parsing failure, through safe fallback output

## UncertaintyEstimator

### Role

Assigns a confidence score to each insight.

### Inputs

- insights
- critiques
- profile data

### Outputs

- `confidence_scores`
- `uncertainty_output`

### Scoring model

It combines four drivers:

1. data quality
2. specificity
3. statistical evidence
4. critic assessment

The first two are rule-based. The last two are LLM-assisted.

### Notable implementation details

- uses batching to reduce LLM latency
- falls back safely when LLM output is invalid or unavailable

## VisualizerAgent

### Role

Generates chart proposals and executable chart code from profile and insight context.

### Inputs

- compact profile context
- insight context
- dataframe column information

### Outputs

- raw LLM output
- validated chart specs
- execution results per chart

### Pipeline stages

1. generate raw visualization proposal
2. parse and validate chart specs
3. execute each chart against the DataFrame
4. serialize result metadata for the UI

### Notable implementation details

- uses the `code` model tier
- chart code is executed in a restricted namespace
- successful figures are persisted as Plotly JSON for later restoration

## ReporterAgent

### Role

Produces the final executive-ready markdown report.

### Inputs

- profile output
- analyst output
- structured insights
- chart summary
- critique summary
- confidence summary
- optional retrieved RAG context

### Outputs

- `reporter_output` returned by the agent
- ultimately stored in graph state as `report_markdown`

### Notable implementation details

- prefers compact digests over large raw markdown where available
- injects prior-run context only when useful
- explicitly incorporates critique and confidence information

## RAGAgent

### Role

Acts as the retrieval bridge to ChromaDB-backed memory.

### Responsibilities

- retrieve prior analysis context
- retrieve filtered insights
- store profiles, insights, reports, and Q&A
- isolate data by `dataset_id`

### Important distinction

`RAGAgent` is not an LLM agent. It is a retrieval and persistence service.

## EvaluationAgent

### Role

Runs the LLM-as-Judge evaluation after the main pipeline finishes.

### Evaluated artifacts

- profiler output
- analyst output
- reporter output

### Notable implementation details

- deterministic validators inject ground truth into prompts
- uses a dedicated evaluation timeout budget
- stores results in typed dataclasses

## Prompt Management

All prompt-driven agents load prompts from a single file:

```text
config/prompts.yaml
```

Prompt access is centralized through:

```text
utils/prompt_loader.py
```

Agents should never compute prompt file paths manually.
