# Configuration

This document explains how configuration works in AutoInsight-AI.

## Configuration Philosophy

The project separates configuration into two categories:

1. static repository paths
2. runtime environment settings

## Static Paths

Static project paths are defined in `config/settings.py` and should not come from `.env`.

Examples:

- `REPO_ROOT`
- `PROMPTS_PATH`
- `DOCS_DIR`
- `DATA_DIR`

These are deterministic repository-relative paths.

## Runtime Settings

Runtime configuration is loaded from environment variables through the frozen `Settings` dataclass in `config/settings.py`.

Main runtime areas:

- LLM provider and models
- timeouts
- profiler knobs
- data-loader behavior
- app title
- LangFuse connectivity
- ChromaDB settings

## `.env` Workflow

Start from:

```bash
cp .env.example .env
```

Then adjust only the runtime variables you need.

## Most Important Variables

### LLM

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `ollama` or `gemini` |
| `OLLAMA_BASE_URL` | Base URL for local Ollama |
| `OLLAMA_LIGHT_MODEL` | Fast model for lightweight tasks |
| `OLLAMA_TEXT_MODEL` | Main reasoning model |
| `OLLAMA_CODE_MODEL` | Model used for chart code generation |
| `GEMINI_MODEL` | Gemini model name when using the Gemini provider |
| `LLM_TIMEOUT` | Timeout for a single LLM request |

### Profiler

| Variable | Description |
|---|---|
| `PROFILER_SAMPLE_ROWS` | Number of sample rows shown and profiled |
| `PROFILER_TOP_VALUES` | Max number of top categorical values retained |
| `PROFILER_DETAIL_MODE` | `fast` or `full` profiler prompt detail |

### Data loading

| Variable | Description |
|---|---|
| `DATA_LOADER_EXCEL_SHEET` | Excel sheet index or name |

### App

| Variable | Description |
|---|---|
| `APP_TITLE` | Streamlit page title |
| `CRITIC_BATCH_SIZE` | Critic batch size |

### Memory

| Variable | Description |
|---|---|
| `OLLAMA_EMBED_MODEL` | Embedding model for ChromaDB retrieval |
| `CHROMA_DIR` | Local persistence directory for ChromaDB |

### LangFuse

| Variable | Description |
|---|---|
| `LANGFUSE_ENABLED` | Enable external tracing |
| `LANGFUSE_BASE_URL` | LangFuse host |
| `LANGFUSE_PUBLIC_KEY` | Project public key |
| `LANGFUSE_SECRET_KEY` | Project secret key |
| `LANGFUSE_ENV` | Environment label |
| `LANGFUSE_RELEASE` | Optional release tag |

### Evaluation

| Variable | Description |
|---|---|
| `EVAL_UNCERTAINTY_HIGH_THRESHOLD` | High-confidence threshold |
| `EVAL_UNCERTAINTY_MEDIUM_THRESHOLD` | Medium-confidence threshold |
| `EVAL_UNCERTAINTY_BATCH_SIZE` | Uncertainty scoring batch size |
| `EVAL_JUDGE_MODEL` | Optional model override for LLM Judge |
| `EVAL_JUDGE_TIMEOUT` | Dedicated judge timeout |

## Model Routing

Model routing is centralized in `utils/llm.py`.

### Tier mapping

| Tier | Default model |
|---|---|
| `light` | `qwen3:4b` |
| `text` | `qwen3:14b` |
| `code` | `qwen2.5-coder:14b` |

### Task mapping

| Task | Tier |
|---|---|
| `profiler` | `light` |
| `analyst` | `text` |
| `critic` | `text` |
| `reporter` | `text` |
| `uncertainty` | `light` |
| `categorizer` | `light` |
| `visualizer` | `code` |

## Prompt Configuration

All agent prompts live in one file:

```text
config/prompts.yaml
```

They are loaded exclusively through:

```text
utils/prompt_loader.load_prompt_section()
```

This keeps prompt ownership centralized and predictable.

## Recommended Local Development Configuration

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LIGHT_MODEL=qwen3:4b
OLLAMA_TEXT_MODEL=qwen3:14b
OLLAMA_CODE_MODEL=qwen2.5-coder:14b
OLLAMA_EMBED_MODEL=nomic-embed-text
LLM_TIMEOUT=300
LANGFUSE_ENABLED=false
```

## Make Targets

Common commands:

| Command | Purpose |
|---|---|
| `make install` | Install dependencies and pre-commit hooks |
| `make run` | Start Streamlit |
| `make run PORT=8502` | Start Streamlit on a different port |
| `make test` | Run fast unit tests |
| `make test-all` | Run the full test suite |
| `make check` | Run the main local quality checks |
| `make langfuse-up` | Start the local LangFuse stack |
| `make langfuse-down` | Stop the local LangFuse stack |
| `make langfuse-reset` | Reset LangFuse data and credentials |

## Configuration Rules

Keep the following rules in mind:

- repository paths do not belong in `.env`
- API keys should never be hardcoded in source files
- prompt paths should never be recomputed inside agents
- model selection should go through `utils/llm.py`
