# Testing

This document describes how the project is validated locally.

## Testing Goals

The project uses tests and quality checks to ensure:

- deterministic utilities behave correctly
- agents handle parsing and fallback logic safely
- orchestration and state flow remain stable
- UI-facing view models return consistent structures
- regressions are caught before changes land

## Test Organization

Tests live in:

```text
tests/
```

They cover multiple layers:

- data loading
- profiling
- agent parsing and formatting
- uncertainty and critic logic
- memory and RAG behavior
- LangFuse integration helpers
- visualization execution
- evaluation subsystem

## Markers

Pytest markers are defined in `pyproject.toml`.

Current markers:

- `smoke`
- `integration`

## Main Commands

### Fast local suite

```bash
make test
```

Equivalent behavior:

```bash
uv run pytest tests/ -v -m "not integration"
```

### Full suite

```bash
make test-all
```

### Lint, format, and test bundle

```bash
make check
```

### CI-style local run

```bash
make ci
```

## What Needs External Services

Most unit tests do not require external services.

Integration-oriented checks may depend on:

- Ollama
- local model availability
- optional LangFuse stack

## Recommended Validation Before a Demo

Before presenting the project:

1. run `make test`
2. run `make check`
3. start the app with `make run`
4. run the full pipeline on one sample dataset
5. optionally verify LangFuse with `make langfuse-up`

## Manual Product Validation

For a product-level sanity check:

- upload `data/sample_sales.csv`
- confirm the pipeline completes
- confirm tabs render content
- confirm diagnostics show node trace
- rerun the same file and verify memory persistence behavior

## What the Tests Do Not Replace

Even with a good test suite, you should still manually check:

- Ollama health and model availability
- Streamlit rendering flow
- full end-to-end latency behavior on your machine
- LangFuse UI integration if enabled

## Best Practice

Use the tests to validate code correctness, and use the sample datasets to validate the actual user experience.
