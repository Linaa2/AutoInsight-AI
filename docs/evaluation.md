# Evaluation

This document explains the three evaluation-related subsystems in the project:

- Critic Agent
- Uncertainty Estimator
- LLM-as-Judge

These subsystems are related, but they do not serve the same purpose.

## Overview

| Component | When it runs | Purpose |
|---|---|---|
| `CriticAgent` | inside the automated pipeline | challenge insights and expose weaknesses |
| `UncertaintyEstimator` | inside the automated pipeline | assign confidence scores to insights |
| `EvaluationAgent` | after the pipeline, on demand | judge output quality across major artifacts |

## 1. Critic Agent

The Critic is part of the analysis pipeline.

It reviews the analyst's insights and produces structured critique objects and markdown summary output.

Main questions it answers:

- is the insight well supported?
- what are its weaknesses?
- what alternative explanations exist?

The Critic improves trustworthiness and helps the reporter qualify strong and weak findings.

## 2. Uncertainty Estimator

The Uncertainty Estimator is also part of the analysis pipeline.

It scores each insight using a hybrid method:

- rule-based drivers
- LLM-based drivers

The result is a confidence score per insight plus a markdown summary.

Its purpose is not to rewrite insights, but to express how reliable they appear.

## 3. LLM-as-Judge

The LLM Judge is a post-run quality evaluation subsystem.

It is not part of the automated LangGraph graph.

It evaluates the quality of three final artifacts:

- profiler output
- analyst output
- reporter output

## LLM Judge Workflow

For each artifact:

1. deterministic validators gather ground truth or structural signals
2. the artifact is truncated safely when needed
3. the judge model evaluates the artifact against a rubric
4. the result is parsed into typed dataclasses

## Rubric-Based Evaluation

Each artifact has a rubric with weighted criteria.

Examples:

- grounding
- completeness
- clarity
- relevance
- actionability
- faithfulness
- coherence

The weighted criteria are combined into:

- an artifact score
- an artifact grade
- suggestions for improvement

## Analyst-Specific Detail

The analyst evaluation is special because it can also produce per-insight breakdowns.

That allows the UI to show:

- overall analyst quality
- per-insight factual correctness
- per-insight relevance
- per-insight actionability

## Data Model

The main evaluation dataclasses live in:

```text
evaluation/schemas.py
```

Key types:

- `EvaluationCriterion`
- `EvaluationResult`
- `AnalystEvaluationResult`
- `InsightScore`
- `PipelineEvaluation`

## Configuration

Evaluation-specific configuration lives in:

```text
evaluation/config.py
```

Important settings:

- uncertainty thresholds
- uncertainty batch size
- judge model override
- judge timeout
- truncation limits
- score-to-grade thresholds

## UI Integration

The completed run exposes two evaluation experiences:

### `Evaluation` tab

Shows:

- critic output
- confidence scores

This reflects what happened inside the automated pipeline.

### `LLM Judge` tab

Shows:

- post-run quality evaluation
- one panel per major artifact
- overall pipeline score and grade

This is user-triggered after the run completes.

## Why the Separation Matters

These components answer different questions:

- Critic: "How should we challenge this insight?"
- Uncertainty: "How confident are we in this insight?"
- LLM Judge: "How good is the final artifact as a product output?"

Keeping them separate avoids mixing:

- analytical reliability
- adversarial review
- output quality scoring

## Operational Notes

- Critic and Uncertainty affect the final report context
- LLM Judge does not alter the main pipeline result
- LLM Judge has its own timeout budget because it is offline evaluation work
