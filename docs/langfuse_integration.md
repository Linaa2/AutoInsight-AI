# LangFuse Integration

LangFuse is an **optional** open-source LLM observability platform that AutoInsight-AI can send traces to.
When enabled, every analysis run is captured as a structured trace showing the full call hierarchy from pipeline entry all the way down to individual LLM tokens.

---

## 1. Why LangFuse?

| Need | How LangFuse helps |
|------|--------------------|
| Debug slow or wrong LLM outputs | See the exact prompt, response, and latency for every generation |
| Compare model versions | Filter traces by `release` or `environment` tag |
| Audit token costs | Token counts appear on every generation span |
| Reproduce a failure | Each run has a stable `trace_id` you can look up later |

---

## 2. What is monitored

Every `run_analysis` / `stream_analysis` call creates **one top-level trace** in LangFuse.
Inside that trace:

```
trace: analysis-run
├── span: profiler         ← ProfilerAgent.describe()
│   └── generation         ← LLM call (captured by CallbackHandler)
├── span: analyst          ← AnalystAgent.run()
│   └── generation
├── span: visualizer       ← run_visualization_pipeline()
│   └── generation
├── span: reporter         ← ReporterAgent.run()
│   └── generation
├── span: rag-context-retrieval   ← memory lookup before run
└── span: rag-storage             ← ChromaDB write after run
```

The `langfuse_trace_id` is also stored in `PipelineState` so you can correlate a Streamlit result with the LangFuse UI.

---

## 3. LangFuse vs internal diagnostics

AutoInsight-AI has **two independent observability layers**; they complement each other:

| Layer | Where it lives | Purpose |
|-------|---------------|---------|
| **Internal diagnostics** (`graph_trace` in `PipelineState`) | In-process Python list | Node-level timing and error details shown directly in the Streamlit UI |
| **LangFuse** (`utils/langfuse_client.py`) | External server | Deep LLM-level tracing: prompts, tokens, latency, multi-run comparison |

Both layers run simultaneously and independently.
Disabling LangFuse has no effect on internal diagnostics.

---

## 4. Environment variables

Add these to your `.env` file (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGFUSE_ENABLED` | `false` | Master switch. Set to `true` to activate. |
| `LANGFUSE_BASE_URL` | `http://localhost:3001` | Canonical LangFuse server URL. Change for cloud: `https://cloud.langfuse.com` |
| `LANGFUSE_HOST` | _(deprecated alias)_ | Backward-compatible alias for `LANGFUSE_BASE_URL` |
| `LANGFUSE_PUBLIC_KEY` | _(empty)_ | Your project's public key |
| `LANGFUSE_SECRET_KEY` | _(empty)_ | Your project's secret key |
| `LANGFUSE_ENV` | `development` | Environment label (`development`, `staging`, `production`) |
| `LANGFUSE_RELEASE` | _(empty)_ | Optional release/version tag (e.g. `v1.2.3` or a git SHA) |

All three conditions must hold for tracing to activate:
- `LANGFUSE_ENABLED=true`
- `LANGFUSE_PUBLIC_KEY` is non-empty
- `LANGFUSE_SECRET_KEY` is non-empty

---

## 5. Self-hosted setup with Docker Compose

Start a local LangFuse instance:

```bash
# Recommended: auto-generates credentials, starts web + worker + storage backends,
# then seeds the missing project membership row required by the UI.
make langfuse-up

# Open the UI
open http://localhost:3001
```

This repo does not require any manual LangFuse setup through the UI:
1. `scripts/setup_langfuse.py` generates `.env.langfuse`
2. LangFuse seeds the org, project, API keys, and admin user on startup
3. The same project keys are synced into `.env`

After `make langfuse-up`, sign in with the credentials printed by the setup script.
If you ran `make langfuse-reset`, open LangFuse in a fresh tab or clear site data for
`localhost:3001` before signing in again because reset rotates the session secret and
project identifiers.

The app connects with:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

To stop:

```bash
docker compose -f docker-compose.langfuse.yml down
```

---

## 6. Enable / disable

**Enable** (development, self-hosted):

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

**Enable** (cloud):

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

**Disable** (default):

```dotenv
LANGFUSE_ENABLED=false
```

The system runs identically with or without LangFuse — all public methods of `LangFuseMonitor` are safe no-ops when disabled.

---

## 7. Trace structure

```
Run id: 9f3a2b1c           (short UUID, also the seed for trace_id)
├── profiler span
│   ├── input:  { "profile_json": "..." }
│   └── generation (LLM)
│       ├── model:  qwen3:14b
│       ├── tokens: { prompt: 312, completion: 89 }
│       └── output: "## Dataset Overview ..."
├── analyst span
│   └── generation (LLM)
├── visualizer span
│   └── generation (LLM)
├── reporter span
│   └── generation (LLM)
├── rag-context-retrieval span (instant)
│   └── output: { "chunks_retrieved": 3 }
└── rag-storage span (instant)
    └── output: { "profile_stored": true, "report_stored": true }
```

The `trace_id` displayed in the Streamlit sidebar links directly to this trace in the LangFuse UI.

---

## 8. Finding a run from Streamlit

After running an analysis with LangFuse enabled:

1. The **Streamlit sidebar** shows:
   - 🟢 LangFuse: enabled — with a link to the host and the trace ID
   - `[🔍 View trace in LangFuse]` — clickable link to the exact trace
2. Copy the **trace ID** shown under the link
3. Paste it in the LangFuse UI search bar to find the run

The `langfuse_trace_id` is also available in the returned `PipelineState` dict under the key `"langfuse_trace_id"`.

---

## 9. Adding tracing to a new agent

New agents only need **two small changes**:

**Step 1 — accept a `callbacks` param:**

```python
def run(self, ..., callbacks: list | None = None) -> dict:
    ...
    result = call_llm_with_messages(system=..., human=..., callbacks=callbacks)
```

**Step 2 — pass callbacks in the graph node:**

```python
# In orchestration/graph.py, inside the new node function:
def new_agent_node(state: PipelineState) -> dict:
    callbacks = lf_monitor.get_llm_callbacks()
    with lf_monitor.node_span("new-agent"):
        result = agent.run(..., callbacks=callbacks)
    return result
```

That's all. No other LangFuse-specific code is needed in agent files.
The `lf_monitor` singleton handles everything and is a no-op when LangFuse is disabled.
