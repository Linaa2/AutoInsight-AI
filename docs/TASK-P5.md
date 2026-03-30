# Phase 5: ChromaDB Memory + RAG Agent

## Overview

Phase 5 adds **persistent contextual memory** to AutoInsight AI using ChromaDB as a vector store, and a **RAG (Retrieval-Augmented Generation) agent** that serves as the bridge between stored context and all other agents.

The RAG agent is **not an LLM agent** — it does not call the LLM itself. It is a **retrieval service** that other agents consume to enrich their prompts with relevant past context.

## Architecture

```
utils/memory.py           → ContextStore: ChromaDB storage + search layer
agents/rag.py             → RAGAgent: retrieval service consumed by other agents
docker-compose.langfuse.yml → LangFuse observability (optional)
tests/test_memory.py      → ContextStore unit tests (require Ollama embeddings)
tests/test_rag.py         → RAGAgent tests (require Ollama embeddings)
```

### Dependency Chain

```
ChromaDB (vector DB)
    ↑
Ollama nomic-embed-text (embedding model)
    ↑
EmbeddingClient (utils/memory.py)
    ↑
ContextStore (utils/memory.py)     ← low-level store/search API
    ↑
RAGAgent (agents/rag.py)           ← high-level retrieval service
    ↑
Other agents (profiler, analyst, reporter, text-to-code, Q&A chat)
```

---

## 1. ContextStore — `utils/memory.py`

### Class: `ContextStore`

Low-level ChromaDB wrapper that handles chunking, embedding, storage, and similarity search.

#### Collections

| Collection | Content | Source Agent |
|---|---|---|
| `profiles` | Dataset profile markdown | Profiler (P1) |
| `insights` | Individual analyst insights | Analyst (P2) |
| `reports` | Executive report markdown | Reporter (P3) |
| `qa_history` | Q&A exchanges (question + answer) | Chat / Q&A |

#### Key Design Decisions

- **Dataset isolation via `dataset_id`**: every stored chunk carries a `dataset_id` metadata field (typically the uploaded filename). This allows multiple datasets to coexist in the same ChromaDB instance without cross-contamination.
- **Structured insight storage**: when insights are passed as `list[dict]`, each insight is stored individually with `priority`, `category`, and `title` as metadata for fine-grained filtered retrieval.
- **Chunking**: uses `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50) to split long texts into embeddable chunks.
- **Embedding model**: `nomic-embed-text` via Ollama (local, no API key needed).

#### Store Methods

| Method | What it stores | Metadata |
|---|---|---|
| `store_profile(text, dataset_id)` | Profiler markdown | source=profiler |
| `store_insights(insights, dataset_id)` | Insight dicts or raw markdown | source=analyst, priority, category, title |
| `store_report(text, dataset_id)` | Reporter markdown | source=reporter |
| `store_qa(question, answer, dataset_id)` | Q&A exchange | source=qa |

#### Search Methods

| Method | Scope | Use Case |
|---|---|---|
| `search(query, collection, dataset_id, k)` | One or all collections | General retrieval |
| `search_insights(query, dataset_id, priority, category, k)` | Insights only | Filtered insight retrieval |
| `search_as_text(query, collection, dataset_id, k)` | One or all collections | Returns joined string (ready for prompt) |

#### Utility Methods

| Method | Description |
|---|---|
| `clear(collection)` | Delete one or all collections |
| `list_collections()` | List existing ChromaDB collections |
| `make_dataset_id(filename)` | Generate a stable dataset_id from a filename |

### Class: `EmbeddingClient`

Factory for embedding models. Mirrors the `LLMClient` pattern from `utils/llm.py`.

```python
embeddings = EmbeddingClient.get_embeddings()
# Returns OllamaEmbeddings(model="nomic-embed-text")
```

### Configuration

| Env Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |

---

## 2. RAGAgent — `agents/rag.py`

### Class: `RAGAgent`

High-level retrieval service that wraps `ContextStore` and provides purpose-specific retrieval methods for other agents.

Each instance is **bound to a `dataset_id`** to isolate context per uploaded file.

#### Retrieval Methods (consumed by other agents)

| Method | Target Consumer | What it retrieves |
|---|---|---|
| `get_context_for_query(query)` | Q&A chat, Text-to-Code | All collections, filtered by dataset |
| `get_context_for_followup(query)` | Q&A chat | Prioritizes insights + Q&A history |
| `get_analysis_context()` | Reporter | Profiles + insights + reports |
| `get_high_priority_insights()` | Reporter, Executive summary | High-priority insights only |
| `get_insights_by_category(category)` | Targeted analysis | Filtered by trend/anomaly/correlation/etc. |

#### Storage Methods (called after pipeline runs)

| Method | When to call |
|---|---|
| `save_analysis(profile_text, insights, report_text)` | After pipeline (profiler → analyst → reporter) completes |
| `save_qa_exchange(question, answer)` | After each Q&A interaction |

#### Utility

| Method | Description |
|---|---|
| `switch_dataset(dataset_id)` | Switch context when user uploads a new file |
| `clear_memory()` | Clear all stored context (full reset) |

#### Built-in Safeguards

- **Deduplication**: duplicate chunks are removed before returning.
- **Truncation**: context is capped at `MAX_CONTEXT_LENGTH` (3000 chars) to avoid overloading LLM prompts. Truncation cuts at the last complete sentence.

---

## 3. Current Status: What Is Built vs. What Is Not Yet Wired

### Built and working

| Component | Status |
|---|---|
| `ContextStore` (utils/memory.py) | Done — stores & retrieves from ChromaDB |
| `RAGAgent` (agents/rag.py) | Done — retrieval service with all methods |
| `EmbeddingClient` | Done — Ollama nomic-embed-text |
| `tests/test_memory.py` | Done — 9 unit tests |
| `tests/test_rag.py` | Done — RAGAgent class defined and tested |

### NOT yet wired (future work)

| Integration Point | Status | What Needs to Happen |
|---|---|---|
| **Pipeline → RAG storage** | Not wired | After the pipeline (profiler → analyst → reporter) completes, call `rag.save_analysis()` to store outputs in ChromaDB |
| **Q&A Chat → RAG retrieval** | Not wired | Before sending a user question to the LLM, call `rag.get_context_for_query()` and inject the result into the prompt |
| **Text-to-Code → RAG** | Not built | `tools/text_to_code.py` is empty. When built, it should use `rag.get_context_for_query()` to provide dataset context for code generation |
| **Reporter → RAG** | Not wired | The Reporter could use `rag.get_analysis_context()` to pull past context instead of receiving everything via function arguments |
| **Streamlit UI → RAG** | Not wired | `app/main.py` should instantiate `RAGAgent(dataset_id=filename)` and call `save_analysis()` after each pipeline step |
| **LangGraph pipeline** | Not built | `graph/pipeline.py` and `graph/state.py` are empty. When built, the RAG agent should be integrated as a node or utility within the graph |

---

## 4. How RAG Will Interact with Other Agents

### RAG does NOT use LangChain agents or LangGraph directly

The RAG agent is a **plain Python class** — it wraps ChromaDB via LangChain's `Chroma` vectorstore and `OllamaEmbeddings`, but it does not use LangChain's agent framework or LangGraph's graph execution.

The interaction model is **service-based**:

```
┌─────────────┐       ┌───────────┐       ┌──────────┐
│  User Query  │──────▶│  RAGAgent  │──────▶│ ChromaDB │
│  (Q&A/Chat)  │       │ .get_*()   │◀──────│ (search) │
└─────────────┘       └─────┬─────┘       └──────────┘
                            │
                     context string
                            │
                            ▼
                    ┌───────────────┐
                    │  LLM Agent    │
                    │ (analyst,     │
                    │  reporter,    │
                    │  text-to-code)│
                    └───────────────┘
```

### Integration Pattern (when wired)

**1. After pipeline completion (storage):**
```python
# In app/main.py or graph/pipeline.py
rag = RAGAgent(dataset_id=uploaded_file.name)
rag.save_analysis(
    profile_text=st.session_state["ai_description"],
    insights=st.session_state["analyst_insights"],
    report_text=st.session_state["reporter_output"],
)
```

**2. Before Q&A / Text-to-Code (retrieval):**
```python
# Enrich the prompt with relevant past context
context = rag.get_context_for_query(user_question)
prompt = f"Context from previous analysis:\n{context}\n\nQuestion: {user_question}"
answer = call_llm_with_messages(system=system_prompt, human=prompt)

# Store the exchange for future retrieval
rag.save_qa_exchange(user_question, answer)
```

**3. For follow-up questions:**
```python
context = rag.get_context_for_followup("the anomalies you mentioned earlier")
# → Prioritizes insights and Q&A history
```

### Will RAG interact with LangGraph?

**Not directly.** When `graph/pipeline.py` is built (Phase 4 / orchestration), there are two possible patterns:

| Pattern | Description | Recommended? |
|---|---|---|
| **RAG as a LangGraph node** | A `rag_storage_node()` function that saves pipeline outputs to ChromaDB after each step | Yes — clean separation |
| **RAG as shared utility** | Each agent node calls `RAGAgent` methods directly within their own node functions | Simpler but less modular |

Example of RAG as a LangGraph node:
```python
def rag_storage_node(state: dict) -> dict:
    """LangGraph node: store pipeline outputs in ChromaDB."""
    rag = RAGAgent(dataset_id=state.get("dataset_id", "default"))
    rag.save_analysis(
        profile_text=state.get("profiler_output", ""),
        insights=state.get("insights", []),
        report_text=state.get("reporter_output", ""),
    )
    return {"rag_stored": True}
```

---

## 5. Testing

```bash
# Memory tests (require Ollama + nomic-embed-text)
uv run python tests/test_memory.py

# RAG tests (require Ollama + nomic-embed-text)
uv run python tests/test_rag.py

# All tests
uv run pytest tests/ -v
```

### Prerequisites

```bash
ollama serve                    # Start Ollama
ollama pull nomic-embed-text    # Download embedding model
```

### ContextStore Tests (`tests/test_memory.py`)

| Test | What it verifies |
|---|---|
| `test_store_profile` | Stores profile text, returns chunk count > 0 |
| `test_store_insights` | Stores insight text, returns chunk count > 0 |
| `test_store_qa` | Stores Q&A exchange |
| `test_store_empty` | Empty/whitespace text returns 0 chunks |
| `test_search_specific_collection` | Searches within a single collection |
| `test_search_all_collections` | Searches across all collections |
| `test_search_as_text` | Returns concatenated string result |
| `test_clear` | Clears a single collection |
| `test_clear_all` | Clears all collections |

---

## 6. Configuration

### Environment Variables (`.env`)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
CHROMA_DIR=./chroma_db
```

### Docker (optional — LangFuse observability)

```bash
docker compose -f docker-compose.langfuse.yml up -d
# LangFuse UI: http://localhost:3001
```

---

## 7. Files Created/Modified

| File | Action |
|---|---|
| `utils/memory.py` | Created — ContextStore + EmbeddingClient |
| `agents/rag.py` | Created — RAGAgent retrieval service |
| `tests/test_memory.py` | Created — 9 ContextStore tests |
| `tests/test_rag.py` | Created — RAGAgent class + tests |
| `docker-compose.langfuse.yml` | Created — LangFuse observability stack |

---

## 8. Next Steps (What Remains)

1. **Wire RAG into Streamlit UI** (`app/main.py`): instantiate `RAGAgent` on file upload, call `save_analysis()` after pipeline steps.
2. **Build Text-to-Code** (`tools/text_to_code.py`): LLM-powered code generation using RAG context for dataset awareness.
3. **Build LangGraph pipeline** (`graph/pipeline.py`, `graph/state.py`): orchestrate all agents with RAG as a storage/retrieval node.
4. **Add Q&A chat tab** in Streamlit: users ask follow-up questions, RAG provides context, responses are stored for future retrieval.
5. **Dataset switching**: call `rag.switch_dataset()` when a new file is uploaded, optionally clear old context.
