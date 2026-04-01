# AutoInsight-AI Documentation

This directory is the canonical documentation set for AutoInsight-AI.

The previous phase notes, task write-ups, and design drafts were archived to:

```text
tmp/docs_archive_2026-04-01/
```

This new documentation is organized to support two goals:

- help an evaluator run and validate the project quickly
- give contributors an accurate picture of the current implementation

## Recommended Reading Paths

### For an evaluator

1. [Quick Start](./quickstart.md)
2. [Streamlit UI](./streamlit-ui.md)
3. [Testing](./testing.md)
4. [Observability](./observability.md) if LangFuse is enabled

### For a contributor

1. [Architecture](./architecture.md)
2. [Pipeline](./pipeline.md)
3. [Agents](./agents.md)
4. [Configuration](./configuration.md)
5. [Developer Guide](./developer-guide.md)

## Documentation Map

| Document | Purpose |
|---|---|
| [quickstart.md](./quickstart.md) | Fastest path to install, run, and verify the project |
| [architecture.md](./architecture.md) | System structure, core modules, and runtime design |
| [pipeline.md](./pipeline.md) | End-to-end LangGraph workflow and state flow |
| [agents.md](./agents.md) | Responsibilities and contracts of every agent |
| [streamlit-ui.md](./streamlit-ui.md) | Product flow, tabs, rendering logic, and user journey |
| [configuration.md](./configuration.md) | Environment variables, model routing, prompts, and Make targets |
| [memory-rag.md](./memory-rag.md) | ChromaDB-backed memory, retrieval, and storage behavior |
| [observability.md](./observability.md) | Diagnostics, LangFuse, traces, spans, and local setup |
| [evaluation.md](./evaluation.md) | Critic, Uncertainty Estimator, and LLM-as-Judge subsystems |
| [testing.md](./testing.md) | Test strategy, commands, and verification checklist |
| [developer-guide.md](./developer-guide.md) | How to extend the system cleanly |

## Current System Summary

AutoInsight-AI is a multi-agent data analysis product built around a single LangGraph workflow:

```text
START
  -> profiler
  -> analyst
  -> critic
  -> uncertainty
  -> visualizer
  -> reporter
  -> rag_storage
END
```

The product is exposed through a Streamlit application, uses local Ollama models by default, stores reusable context in ChromaDB, and can optionally send run traces to a self-hosted LangFuse stack.
