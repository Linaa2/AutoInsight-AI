# Quick Start

This guide is the fastest reliable path to run AutoInsight-AI locally and verify that the project works end to end.

## Goal

At the end of this guide, you should be able to:

- launch the Streamlit application
- upload a sample dataset
- run the full analysis pipeline
- inspect the generated profile, insights, charts, report, and diagnostics

## Prerequisites

- Python `3.11` or `3.12`
- `uv`
- Ollama installed locally
- enough RAM to run the selected Ollama models

## 1. Install dependencies

From the project root:

```bash
uv sync --extra dev
```

You do not need to activate a virtual environment manually if you use `uv run` and the provided `make` targets.

## 2. Configure the environment

Copy the template:

```bash
cp .env.example .env
```

For the default local setup, use:

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

## 3. Pull the required models

```bash
ollama pull qwen3:4b
ollama pull qwen3:14b
ollama pull qwen2.5-coder:14b
ollama pull nomic-embed-text
```

If Ollama is not already running:

```bash
ollama serve
```

## 4. Launch the app

```bash
make run
```

Default URL:

```text
http://localhost:8501
```

To use another free port:

```bash
make run PORT=8502
```

Good alternatives are usually `8502`, `8503`, `8601`, and `9001`.

## 5. Run a full analysis

Use one of the bundled datasets:

- `data/sample_sales.csv`
- `data/sample_saas_health.csv`

In the app:

1. Upload the dataset
2. Review the dataset preview
3. Click `Run Full Analysis`
4. Wait for the pipeline to complete

## 6. What to verify

The run is considered successful if the following appear:

- `Profile` tab with the generated markdown profile
- `Insights` tab with structured findings
- `Evaluation` tab with critique and confidence information
- `Visualizations` tab with at least one rendered Plotly chart
- `Report` tab with the final markdown report
- `Diagnostics` tab with node execution trace

## Optional: enable LangFuse

If you also want GenAI tracing:

```bash
make langfuse-up
```

Then set or confirm in `.env`:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://localhost:3001
```

LangFuse UI:

```text
http://localhost:3001
```

## Fast validation checklist

- App starts without import errors
- File upload works for CSV, Excel, or Parquet
- Pipeline status advances through all nodes
- Final report is generated
- Diagnostics show node timings
- Re-running the same file shows memory persistence behavior
