# Developer Guide

This guide explains how to extend AutoInsight-AI without breaking its architectural boundaries.

## Core Rules

Follow these rules when adding functionality:

- keep orchestration in `orchestration/`
- keep prompts in `config/prompts.yaml`
- keep runtime configuration in `config/settings.py` and `.env`
- keep deterministic logic out of agent prompt code
- keep Streamlit focused on rendering and user actions
- keep LangFuse access centralized in `utils/langfuse_client.py`

## Adding a New Agent

To add a new pipeline agent cleanly:

### 1. Create the agent

Add a new module under `agents/`.

The agent should:

- accept structured inputs
- return structured outputs
- load prompts via `utils/prompt_loader.py`
- obtain models through `utils/llm.py`

### 2. Add prompts

Create a new top-level section in:

```text
config/prompts.yaml
```

### 3. Extend the state contract

Add any new state keys to:

```text
orchestration/state.py
```

### 4. Add the node function

Implement a node function in:

```text
orchestration/graph.py
```

It should:

- read only the keys it needs
- write only the keys it produces
- append a trace entry
- fail gracefully

### 5. Wire the graph

Register the node in `build_graph()` and add the required edges.

## Adding a New Prompted Feature Outside the Graph

If the new feature is post-run or auxiliary:

- keep it out of the automated graph
- attach it to the UI after the main result exists
- store its state separately when needed

This is how the current `LLM Judge` feature is integrated.

## Adding Observability

For a new traced node:

- wrap node execution in `lf_monitor.node_span(...)`
- pass `lf_monitor.get_llm_callbacks()` into LLM calls
- write a `NodeTraceEntry`

Do not import the LangFuse SDK directly outside `utils/langfuse_client.py`.

## Adding Memory-Aware Features

If a feature needs retrieval context:

- use `RAGAgent` rather than calling `ContextStore` directly from the UI
- keep retrieval scoped by `dataset_id`
- prefer best-effort behavior

## Updating the UI

If you add a new user-facing result:

- render it through `app/main.py`
- use view-model helpers in `app/*_view.py` when appropriate
- keep business logic out of the UI layer

## Updating Documentation

The documentation in `docs/` is now the canonical documentation set.

When you change architecture, workflow, or setup:

1. update the relevant file in `docs/`
2. keep `docs/README.md` consistent
3. update the root `README.md` if the evaluator path changed

## What to Avoid

Avoid these anti-patterns:

- recomputing prompt paths inside agents
- using raw `os.getenv()` everywhere instead of centralized settings
- mixing business logic into Streamlit rendering code
- bypassing the shared state contract
- duplicating model-routing logic across modules

## Best Mental Model

The clean extension path is:

```text
Prompt
  -> agent
  -> state key additions
  -> graph node
  -> UI rendering
  -> diagnostics and docs update
```

If a change does not fit this flow cleanly, it is usually a sign that the responsibility boundary should be reconsidered.
