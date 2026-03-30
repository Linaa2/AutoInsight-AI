# 📊 Data Analyst Agent — Technical Specification (POC)

## 1. Overview

### 1.1 Objective

Build a **local-first, agentic data analysis assistant** that:
- ingests tabular datasets (CSV/XLSX)
- performs deterministic profiling
- generates insights, charts, and a report
- supports conversational Q&A via text-to-code execution
- **audits its own outputs (adversarial critique)**
- **communicates uncertainty explicitly**

---

### 1.2 Key Innovation

> This system does not only generate analysis — it **critiques its own assumptions** and **quantifies uncertainty**.

---

### 1.3 Core Capabilities

| Capability | Description |
|----------|------------|
| Profiling | Deterministic dataset analysis |
| Insights | LLM-generated structured insights |
| Visualization | Automatic chart generation |
| Reporting | Business-oriented summary |
| Q&A | Text-to-code execution over dataset |
| Critique | Self-review of weak assumptions |
| Uncertainty | Confidence estimation & caveats |
| Memory (RAG) | Contextual retrieval of prior analysis |

---

# 2. Design Principles

## 2.1 Deterministic First, LLM Second
- All numeric/statistical computation → **pandas**
- LLM → interpretation, explanation, formatting

---

## 2.2 Local-First Architecture
- Primary inference via **Ollama**
- Optional cloud fallback (OpenRouter) for difficult tasks

---

## 2.3 Modular Agent Design
Each agent has:
- clear responsibility
- structured inputs/outputs
- no overlapping logic

---

## 2.4 Structured Outputs
- JSON for charts
- Markdown for reports
- Typed objects for pipeline state

---

## 2.5 Safe Execution
- restricted code execution environment (“sandbox”)
- controlled libraries and runtime

---

# 3. System Architecture

## 3.1 High-Level Flow

```text
Upload Dataset
   ↓
Deterministic Profiling (pandas)
   ↓
Profiler Agent
   ↓
Analyst Agent
   ↓
┌───────────────┬────────────────┐
│               │                │
Visualizer   Critic        Uncertainty
│               │                │
└───────────────┴────────────────┘
           ↓
        Reporter
           ↓
        UI Output
           ↓
        Q&A (Text-to-Code + RAG)
```

---

## 3.2 Logical Components

| Layer         | Technology     | Role                      |
| ------------- | -------------- | ------------------------- |
| UI            | Streamlit      | User interaction          |
| Data          | pandas         | Deterministic computation |
| Orchestration | LangGraph      | Agent coordination        |
| LLM           | Ollama         | Local inference           |
| Memory (RAG)  | ChromaDB       | Context retrieval         |
| Observability | LangFuse       | Tracing & metrics         |
| Execution     | Python sandbox | Safe code execution       |

---

# 4. Infrastructure

## 4.1 Local Setup

* Python environment
* Ollama running locally
* Streamlit app
* ChromaDB (local persistence)
* Optional LangFuse (cloud or disabled)

---

## 4.2 Architecture Diagram

```text
User
 ↓
Streamlit UI
 ↓
Python Backend
 ├─ pandas
 ├─ LangGraph
 ├─ Sandbox Executor
 ├─ ChromaDB
 └─ LLM Client
        ↓
     Ollama
```

---

# 5. Data Layer

## 5.1 Supported Inputs

* CSV
* XLSX

---

## 5.2 Deterministic Profiling

Computed using pandas:

* shape (rows, columns)
* column types
* missing values
* duplicates
* descriptive statistics
* value distributions
* sample rows

---

## 5.3 Output Schema

```json
{
  "shape": [1000, 12],
  "columns": {...},
  "missing": {...},
  "stats": {...},
  "samples": [...]
}
```

---

# 6. Agent Design

## 6.1 Profiler Agent

### Input

* deterministic profile

### Output

* structured markdown description

### Role

* explain structure
* highlight data quality issues

---

## 6.2 Analyst Agent

### Input

* profile
* samples

### Output

```json
{
  "insights": [
    {
      "title": "...",
      "observation": "...",
      "hypothesis": "...",
      "recommendation": "..."
    }
  ]
}
```

---

## 6.3 Visualizer Agent

### Input

* schema
* insights

### Output (strict JSON)

```json
{
  "charts": [
    {
      "title": "...",
      "type": "bar",
      "code": "plotly code"
    }
  ]
}
```

---

## 6.4 Critic Agent (🔥 NEW)

### Purpose

Attack assumptions and identify weaknesses.

### Input

* insights
* profile

### Output

```json
{
  "fragile_insights": [...],
  "unsupported_claims": [...],
  "alternative_explanations": [...],
  "visualization_warnings": [...]
}
```

---

## 6.5 Uncertainty Estimator (🔥 NEW)

### Purpose

Estimate reliability of outputs.

### Input

* profile
* insights
* critique
* execution status

### Output

```json
{
  "overall_confidence": "medium",
  "score": 0.65,
  "drivers": [...],
  "safe": [...],
  "caution": [...]
}
```

---

## 6.6 Reporter Agent

### Input

* profile
* insights
* charts
* critique
* uncertainty

### Output

Markdown report:

```markdown
## Summary
...

## Key Insights
...

## Critique
...

## Confidence
...

## Recommendations
...
```

---

# 7. Text-to-Code (Q&A)

## 7.1 Purpose

Answer user queries by generating and executing pandas code.

---

## 7.2 Flow

```text
User Question
   ↓
Prompt → Code Generation
   ↓
Sandbox Execution
   ↓
Result + Explanation
```

---

## 7.3 Example

Input:

> "Top 5 products by revenue"

Generated:

```python
df.groupby("product")["revenue"].sum().nlargest(5)
```

---

# 8. Sandbox (Execution Layer)

## 8.1 Definition

A **restricted execution environment** that:

* runs LLM-generated code safely
* prevents system access

---

## 8.2 Allowed

* pandas
* numpy
* plotly
* safe builtins

---

## 8.3 Forbidden

* file I/O
* OS access
* network calls
* subprocess

---

## 8.4 Constraints

* timeout (e.g. 5s)
* limited memory
* isolated namespace

---

# 9. RAG Design

## 9.1 Principle

RAG is used for:

> retrieving prior analysis context, not raw dataset rows

---

## 9.2 Data stored

* profile output
* insights
* reports
* Q&A history

---

## 9.3 Optional knowledge base

* data analysis best practices
* visualization rules

---

## 9.4 Retrieval usage

Used in:

* Q&A
* report generation
* follow-up queries

---

# 10. Model Strategy

## 10.1 Primary: Ollama (local)

Used for:

* profiling explanation
* insights
* reporting
* embeddings

---

## 10.2 Optional: second local model

Used for:

* code generation
* structured outputs

---

## 10.3 Optional fallback: OpenRouter

Used only if:

* code generation fails
* JSON output unreliable

---

## 10.4 Routing logic

```text
Default → Ollama
If failure → fallback model
```

---

# 11. Observability & Evaluation

## 11.1 LangFuse

Track:

* prompts
* outputs
* latency
* token usage

---

## 11.2 Metrics

* code execution success rate
* chart generation success
* latency per agent
* failure rates

---

## 11.3 Functional evaluation

Test with:

* 5–10 benchmark questions
* known expected answers

---

# 12. Risks & Mitigations

| Risk                  | Mitigation               |
| --------------------- | ------------------------ |
| hallucinated insights | deterministic profiling  |
| invalid JSON          | strict schema + retries  |
| code failure          | sandbox + validation     |
| weak model            | fallback or second model |
| unsafe execution      | restricted sandbox       |
| context overload      | RAG retrieval            |

---

# 13. Implementation Roadmap

## Phase 1 (MVP)

* dataset upload
* profiling
* profiler + analyst + reporter

---

## Phase 2

* visualizer
* basic charts

---

## Phase 3

* text-to-code Q&A
* sandbox

---

## Phase 4 (🔥 differentiation)

* critic agent
* uncertainty estimator

---

## Phase 5 (optional)

* RAG memory
* evaluation dashboards

---

# 14. Project Structure

```text
project/
├── app.py
├── agents/
├── data/
├── orchestration/
├── chat/
├── rag/
├── evaluation/
├── llm/
└── prompts/
```

---

# 15. Final Positioning

> **Data Analyst Agent** is a local-first, multi-agent GenAI system that combines deterministic computation, LLM reasoning, adversarial critique, and uncertainty estimation to deliver robust and trustworthy data analysis.

---

# 16. Key Differentiators

* deterministic + LLM hybrid
* agentic architecture
* text-to-code execution
* adversarial self-critique
* uncertainty-aware outputs
* local-first + optional cloud fallback

---

# 17. Final Recommendation

Focus on:

* stability first
* simplicity of agents
* strong prompts
* minimal but meaningful critique & uncertainty

Avoid:

* over-engineering
* too many agents
* overly complex sandbox

---

# 🚀 End Goal

A system that:

* works reliably
* is easy to demo
* is technically sound
* clearly stands out