# Configuration Architecture

AutoInsight-AI separates configuration into two layers:

1. **Static project paths** — deterministic, repository-relative, defined in code.
2. **Runtime settings** — env-driven, vary per developer / deployment.

---

## What belongs where

| Category | Where it lives | Examples |
|---|---|---|
| Prompts file path | `config/settings.py` (static) | `PROMPTS_PATH` |
| Repo root / docs dir | `config/settings.py` (static) | `REPO_ROOT`, `DOCS_DIR` |
| LLM model names | `.env` → `config/settings.py` (runtime) | `OLLAMA_TEXT_MODEL` |
| Provider & timeout | `.env` → `config/settings.py` (runtime) | `LLM_PROVIDER`, `LLM_TIMEOUT` |
| Profiler knobs | `.env` → `config/settings.py` (runtime) | `PROFILER_SAMPLE_ROWS` |
| API keys | `.env` only (never in code) | `GOOGLE_API_KEY` |

**Rule**: if it's a filesystem path internal to the repo, it's **static**.
If different contributors may want different values, it's **runtime** (`.env`).

---

## `config/settings.py`

Single source of truth. Two parts:

### Static paths (module-level constants)

```python
from config.settings import REPO_ROOT, PROMPTS_PATH, DOCS_DIR, DATA_DIR
```

These are computed once from `Path(__file__)`. No environment variable involved.

### Runtime settings (frozen dataclass)

```python
from config.settings import settings

settings.OLLAMA_TEXT_MODEL   # "qwen3:14b"
settings.LLM_TIMEOUT         # 60
settings.PROFILER_SAMPLE_ROWS # 5
```

`Settings` is a frozen `@dataclass` that reads from `os.getenv()` at import
time. `.env` is loaded via `python-dotenv` at the top of the module.

---

## Prompt loading

Prompts live in `config/prompts.yaml`. The path is `config.settings.PROMPTS_PATH`.

All agents use a single shared loader:

```python
from utils.prompt_loader import load_prompt_section

prompts = load_prompt_section("profiler")   # {"system": "...", "human": "..."}
prompts = load_prompt_section("visualizer")
```

Agents **never** compute their own path to prompts. If the section key is
missing, a `KeyError` is raised immediately.

---

## LLM model routing

`utils/llm.py` provides `LLMClient`:

```python
from utils.llm import LLMClient

llm = LLMClient.get_text_llm()   # OLLAMA_TEXT_MODEL → qwen3:14b
llm = LLMClient.get_code_llm()   # OLLAMA_CODE_MODEL → qwen2.5-coder:14b
```

Model names come from `config.settings.settings`. The provider (`ollama` or
`gemini`) selects which LangChain adapter is instantiated.

---

## `.env.example`

```
LLM_PROVIDER=ollama
OLLAMA_TEXT_MODEL=qwen3:14b
OLLAMA_CODE_MODEL=qwen2.5-coder:14b
LLM_TIMEOUT=60
```

Copy to `.env` and adjust. The following do **NOT** belong in `.env`:

- `PROMPTS_PATH` — it's a repo path, defined in `config/settings.py`.
- Any hardcoded filesystem path.

---

## Anti-patterns removed in this refactor

| Before | After |
|---|---|
| `Path(__file__).parent.parent / "config" / ...` in agents | `config.settings.PROMPTS_PATH` (static constant) |
| `os.getenv("PROMPTS_PATH", ...)` | Removed — not a runtime setting |
| `sys.path.insert(0, ...)` in Streamlit app | Removed — `uv run` sets CWD correctly |
| Raw `os.getenv()` scattered in tools | `config.settings.settings.*` centralized |
| `monkeypatch.setenv()` in tests for tool config | Constructor parameters with settings-based defaults |
