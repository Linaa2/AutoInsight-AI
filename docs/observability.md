# Observability

AutoInsight-AI has two observability layers:

1. built-in local diagnostics
2. optional LangFuse tracing

They complement each other and are designed to work independently.

## 1. Built-in Diagnostics

Internal diagnostics are always part of the project.

They are driven by:

- `graph_trace` in `PipelineState`
- `memory_trace` for RAG activity
- telemetry helpers in `diagnostics/telemetry.py`
- diagnostics rendering in `diagnostics/renderer.py`

## What diagnostics capture

Per node, the system records:

- node name
- status
- start and finish timestamps
- duration
- keys read and written
- summary
- error string when relevant
- model and resource telemetry when available

## Diagnostics in the UI

The `Diagnostics` tab shows:

- the final pipeline diagram
- aggregate metrics
- per-node detail cards
- memory and RAG activity
- downloadable trace JSON

## 2. LangFuse

LangFuse is the optional external GenAI observability layer.

When enabled, one analysis run becomes one top-level trace.

Inside that trace, the system can create:

- node spans
- event spans
- model generations

## Trace Structure

A typical run looks like this conceptually:

```text
trace: analysis-run
  -> span: profiler
  -> span: analyst
  -> span: critic
  -> span: uncertainty
  -> span: visualizer
  -> span: reporter
  -> event: rag-context-retrieval
  -> event: rag-storage
```

The exact trace detail depends on:

- whether LangFuse is enabled
- whether callbacks are attached to the LLM calls
- whether authentication succeeds for the current run

## LangFuse Integration Points

Main integration module:

```text
utils/langfuse_client.py
```

Main lifecycle:

1. `begin_run()` creates a top-level trace
2. each graph node can run inside `node_span()`
3. `get_llm_callbacks()` enables LangChain generation tracing
4. `log_event()` records discrete events such as memory operations
5. `end_run()` and `flush()` finalize the run

## Self-Hosted Local Setup

Local LangFuse is provided through:

```text
docker-compose.langfuse.yml
```

The local stack includes:

- LangFuse web
- LangFuse worker
- PostgreSQL
- ClickHouse
- Redis
- MinIO

### Start it

```bash
make langfuse-up
```

### Stop it

```bash
make langfuse-down
```

### Reset it

```bash
make langfuse-reset
```

## Credential Management

Credential generation is automated by:

```text
scripts/setup_langfuse.py
```

It:

- generates `.env.langfuse`
- seeds the local LangFuse stack
- syncs project keys into `.env`

This avoids manual project and key creation in the UI.

## LangFuse Configuration

Required `.env` values:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://localhost:3001
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

## Streamlit Surfaces

When enabled, LangFuse appears in:

- the sidebar status block
- the Observability tab
- trace id links after completed runs

## Diagnostics vs LangFuse

| Capability | Local diagnostics | LangFuse |
|---|---|---|
| Node timing | yes | yes |
| Resource telemetry | yes | indirect |
| Prompt and response tracing | no | yes |
| Cross-run browsing | limited | yes |
| External run history | no | yes |

## Failure Behavior

Observability must not break the product.

If LangFuse is unavailable or misconfigured:

- the application should keep running
- local diagnostics still work
- LangFuse should degrade to a no-op for that run

## Practical Use

Use diagnostics when you want:

- fast local visibility into node behavior
- resource and latency information in the app

Use LangFuse when you want:

- prompt and completion inspection
- run-to-run comparison
- external trace browsing
- debugging of LLM behavior
