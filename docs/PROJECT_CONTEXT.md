# AutoInsight AI — Project Context for Presentation

> Système multi-agent d'analyse de données automatique
> **Stack** : LangGraph · Ollama · Streamlit · ChromaDB · LangFuse

---

## 1. Vue d'ensemble du projet

**AutoInsight AI** est un assistant d'analyse de données basé sur une architecture **multi-agent**. Il combine du **profilage déterministe** (pandas) avec des **agents LLM spécialisés** pour générer automatiquement un rapport exécutif complet à partir d'un simple fichier CSV ou Excel.

### Chiffres clés

| Métrique | Valeur |
|----------|--------|
| Lignes de code (Python) | ~9 700 |
| Fichiers source | 59 |
| Tests unitaires | 180 |
| Tests d'intégration (LLM) | 7 |
| Commits | 33 |
| Agents dans le pipeline | 6 + RAG storage |
| Phases de développement | 8 (P0 → P8) |

### Équipe

| Membre | Email | Rôle principal |
|--------|-------|----------------|
| ELAMINE Mohammed | elamine.mohammed.14@gmail.com | — |
| RHIATI HAZIME Lina | lina.rhiati2@gmail.com | — |
| BOUTROUFT Younes | bft.younes@gmail.com | — |

---

## 2. Stack technique

| Composant | Outil | Rôle |
|-----------|-------|------|
| **Orchestration** | LangGraph (StateGraph) | Pipeline multi-agent, flux de données, streaming |
| **LLM local** | Ollama + Mistral 7B / Qwen3 14B | Génération de texte (insights, critique, rapport) |
| **LLM code** | Ollama + Qwen2.5-coder 14B | Génération de code Plotly pour les charts |
| **LLM cloud (fallback)** | Google Gemini 1.5 Flash | Fallback si Ollama indisponible |
| **UI** | Streamlit | Interface utilisateur progressive |
| **Mémoire vectorielle** | ChromaDB + nomic-embed-text | RAG — contexte des analyses précédentes |
| **Observabilité** | LangFuse (Docker) | Tracing des appels LLM, métriques |
| **Visualisation** | Plotly Express + Graph Objects | Charts interactifs |
| **Data processing** | Pandas + NumPy | Profilage déterministe |
| **Qualité** | Ruff, mypy, pytest, pre-commit | Lint, types, tests, hooks |
| **Packaging** | uv + hatchling | Gestion dépendances et build |

---

## 3. Architecture du pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
│  Upload CSV/Excel → Progressive tabs → Download reports        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   LangGraph     │
                    │   StateGraph    │
                    └────────┬────────┘
                             │
     ┌───────────────────────┼───────────────────────────┐
     │                       │                           │
     ▼                       ▼                           ▼
┌─────────┐          ┌──────────┐                ┌──────────┐
│ Profiler │─────────▶│ Analyst  │───────────────▶│  Critic  │
│(détérm.) │          │ (LLM)    │                │  (LLM)   │
└─────────┘          └──────────┘                └────┬─────┘
                                                      │
                                                      ▼
┌───────────┐         ┌──────────┐           ┌──────────────┐
│ Visualizer│◀────────│ Reporter │◀──────────│ Uncertainty  │
│   (LLM)   │─────────▶│  (LLM)  │           │ (Rules+LLM)  │
└───────────┘         └────┬─────┘           └──────────────┘
                           │
                    ┌──────▼──────┐
                    │ RAG Storage │
                    │ (ChromaDB)  │
                    └─────────────┘
```

### Ordre d'exécution

```
START → profiler → analyst → critic → uncertainty → visualizer → reporter → rag_storage → END
```

Chaque nœud lit et écrit dans un **PipelineState** (TypedDict) partagé. Le streaming LangGraph permet un **rendu progressif** : chaque tab apparaît dès que l'agent correspondant termine.

---

## 4. Les 6 agents

### 4.1 📊 Profiler Agent

| | |
|---|---|
| **Fichier** | `agents/profiler.py` + `tools/profiler_engine.py` |
| **Input** | DataFrame (via `df_dict`) |
| **Output** | `profile_data` (dict JSON), `profile_markdown` (str) |
| **Méthode** | Profilage **déterministe** (pandas) + interprétation LLM |

**Ce qu'il fait** :
- **DataProfiler** (pur pandas, pas de LLM) : calcule shape, types, stats numériques (mean, median, std, skew, kurtosis), missing values, duplicates, top values, sample rows
- **ProfilerAgent** (LLM) : prend le profil JSON et génère une description markdown narrative

**Métriques calculées** : lignes, colonnes, doublons, % missing, mémoire, dtype categories, skewness, kurtosis, zeros count, datetime ranges

### 4.2 💡 Analyst Agent

| | |
|---|---|
| **Fichier** | `agents/analyst.py` |
| **Input** | `profile_markdown`, `profile_data`, `df_dict` |
| **Output** | `insights` (list[dict]), `insights_markdown` (str) |
| **Méthode** | LLM → JSON structuré → validation → catégorisation |

**Ce qu'il fait** :
- Génère 3-6 insights structurés à partir du profil
- Chaque insight : `title`, `observation`, `hypothesis`, `recommendation`, `priority`, `category`
- **InsightCategorizer** : classification par mots-clés (rapide) ou LLM (précise)
- **InsightFormatter** : conversion en markdown groupé par catégorie

**Catégories** : trend 📈, anomaly ⚠️, correlation 🔗, distribution 📊, general 💡
**Priorités** : high 🔴, medium 🟡, low 🟢

**Validation** : `extract_json()` parse le JSON bruité du LLM, `validate_insight()` vérifie les champs requis, `normalize_priority()` normalise les valeurs, `fallback_parse_markdown()` en cas d'échec JSON

### 4.3 🔎 Critic Agent

| | |
|---|---|
| **Fichier** | `agents/critic.py` |
| **Input** | `insights`, `profiler_output`, `profile_data` |
| **Output** | `critiques` (list[dict]), `critic_output` (str markdown) |
| **Méthode** | LLM adversarial — 1 appel par insight |

**Ce qu'il fait** :
- **Revue adversariale** de chaque insight de l'Analyst
- Identifie les forces, faiblesses, et hypothèses alternatives
- Attribue un **verdict** et un niveau de **confiance**

**Structure de sortie (par insight)** :
```json
{
  "insight_title": "Sales concentrated in Ile-de-France",
  "strengths": "Backed by data: 480/1500 orders (32%)",
  "weaknesses": "Doesn't account for population density",
  "alternatives": "Could reflect marketing spend or warehouse location",
  "confidence": "medium",
  "verdict": "partially_supported"
}
```

**Verdicts** : `supported` ✅ | `partially_supported` ⚠️ | `weak` ❌

### 4.4 🎯 Uncertainty Estimator

| | |
|---|---|
| **Fichier** | `agents/uncertainty.py` |
| **Input** | `insights`, `critiques`, `profile_data` |
| **Output** | `confidence_scores` (list[dict]), `uncertainty_output` (str markdown) |
| **Méthode** | Hybride — règles déterministes + LLM |

**Ce qu'il fait** :
- Score de confiance **0-100%** pour chaque insight
- 4 drivers (0-25 pts chacun) :

| Driver | Méthode | Ce qu'il évalue |
|--------|---------|-----------------|
| 📦 Data Quality | Règles | Taille du dataset, missing values, duplicates |
| 🔬 Specificity | Règles | Noms de colonnes, chiffres cités, recommandation actionnable |
| 📐 Statistical Evidence | LLM | Solidité des observations, effect size, tests mentionnés |
| 🧐 Critic Assessment | LLM | Sévérité des critiques du Critic Agent |

**Niveaux** : High 🟢 (80-100%) | Medium 🟡 (50-79%) | Low 🔴 (0-49%)

**Contrat Critic→Uncertainty** : le lien se fait par `insight_title` (exact match). Le Critic copie le titre sans reformulation.

### 4.5 📈 Visualizer Agent

| | |
|---|---|
| **Fichier** | `agents/visualizer.py` + `visualization/` |
| **Input** | `df_dict`, `profile_markdown`, `insights_markdown` |
| **Output** | `visualization_result` (dict avec charts) |
| **Méthode** | LLM code → parse → sandbox exec |

**Ce qu'il fait** :
- Le LLM génère des spécifications de charts en JSON (titre, type, code Plotly)
- **Parser** (`visualization/parser.py`) : valide le JSON, extrait les `ChartSpec`
- **Executor** (`visualization/executor.py`) : exécute le code dans un **sandbox restreint**

**Sandbox** :
```python
namespace = {
    "__builtins__": {},  # Aucun builtin → pas d'import, pas d'exec
    "df": df, "pd": pandas, "px": plotly.express,
    "go": plotly.graph_objects, "np": numpy,
}
exec(code, namespace)
fig = namespace.get("fig")
```

**Types de charts** : bar, scatter, histogram, line, box, heatmap, pie, treemap, sunburst, funnel, area

**Modèle routing** : utilise `OLLAMA_CODE_MODEL` (qwen2.5-coder:14b) au lieu du modèle texte

### 4.6 📄 Reporter Agent

| | |
|---|---|
| **Fichier** | `agents/reporter.py` |
| **Input** | Tous les outputs précédents |
| **Output** | `report_markdown` (str) |
| **Méthode** | LLM — synthèse narrative |

**Ce qu'il fait** :
- Synthétise **tous** les outputs en un rapport exécutif markdown
- Intègre dynamiquement le **Critic Review** et les **Confidence Scores** s'ils existent

**Sections du rapport** :
1. 📝 Executive Summary
2. 📊 Dataset Description
3. 🔍 Key Insights (enrichi par critic + uncertainty)
4. 📈 Visualizations
5. ✅ Recommendations
6. ⚠️ Limitations & Next Steps

**Injection conditionnelle** :
- Si `critic_output` existe → section `## 🔎 Critic Review` ajoutée au prompt
- Si `uncertainty_output` existe → section `## 🎯 Insight Confidence Scores` ajoutée au prompt
- Le LLM qualifie les insights low/medium confidence dans le rapport final

---

## 5. Mémoire et RAG (ChromaDB)

| | |
|---|---|
| **Fichier** | `agents/rag.py` + `utils/memory.py` |
| **Base vectorielle** | ChromaDB (local, persisté dans `./chroma_db`) |
| **Embeddings** | Ollama nomic-embed-text |

**Fonctionnement** :
1. À la fin de chaque analyse, `rag_storage_node` stocke le profil, les insights et le rapport dans ChromaDB
2. Chaque dataset a un `dataset_id` unique (hash du nom de fichier)
3. Lors d'une nouvelle analyse, le Reporter reçoit le **contexte des analyses précédentes** via RAG retrieval
4. Isolation par `dataset_id` : pas de contamination entre datasets

**Données stockées** : profil markdown, insights structurés, rapport complet, métadonnées (timestamp, file name)

---

## 6. Observabilité — LangFuse

| | |
|---|---|
| **Fichier** | `utils/langfuse_client.py` |
| **Déploiement** | Docker Compose (`docker-compose.langfuse.yml`) |
| **Interface** | `http://localhost:3001` |

**Ce qui est tracé** :
- Chaque **run** d'analyse = 1 trace LangFuse
- Chaque **agent** = 1 span dans la trace
- Chaque **appel LLM** = callbacks LangChain capturés
- Événements RAG (stockage/retrieval)

**Singleton pattern** : `LangFuseMonitor` avec dégradation gracieuse — si LangFuse est désactivé ou inaccessible, toutes les méthodes retournent des valeurs vides (jamais d'erreur).

**Configuration** (`.env`) :
```
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

---

## 7. Interface utilisateur (Streamlit)

### 7.1 Flux utilisateur

1. **Upload** : glisser-déposer un CSV ou Excel dans la sidebar
2. **Lancement** : bouton "Run analysis"
3. **Rendu progressif** : les tabs apparaissent au fur et à mesure que chaque agent termine
4. **Exploration** : navigation entre les tabs avec téléchargement possible

### 7.2 Tabs de l'interface

| Tab | Contenu |
|-----|---------|
| 📊 Profile | KPIs (rows, cols, duplicates, missing %, memory) + détail colonnes + échantillon |
| 💡 Insights | Résumé par priorité + par catégorie + cards interactives par insight |
| 🔎 Evaluation | **Critic Review** (verdicts, forces/faiblesses) + **Confidence Scores** (progress bars, 4 drivers) |
| 📈 Visualizations | Charts Plotly interactifs + code source + explication |
| 📄 Report | Rapport markdown complet + téléchargement |
| 🧠 Memory | Historique RAG, insights prioritaires, insights par catégorie |
| 🔧 Diagnostics | Pipeline diagram (React Flow), métriques par agent (CPU, RAM, durée, tokens), trace JSON |

### 7.3 Rendu progressif

Pendant l'exécution, chaque tab montre :
- ✓ quand l'agent a terminé
- ⏳ quand l'agent est en cours
- "Waiting…" pour les agents pas encore lancés

Le tab **Evaluation** combine critic + uncertainty : il montre ⏳ tant que l'un des deux tourne, et ✓ seulement quand les deux sont terminés.

---

## 8. PipelineState — Contrat de données

```python
class PipelineState(TypedDict, total=False):
    # Inputs
    df_dict: list[dict[str, Any]]
    file_name: str
    dataset_id: str

    # Profiler
    profile_data: dict[str, Any]
    profile_markdown: str

    # Analyst
    insights: list[dict[str, Any]]
    insights_markdown: str

    # Critic (Phase 8)
    critiques: list[dict[str, Any]]
    critic_output: str

    # Uncertainty (Phase 8)
    confidence_scores: list[dict[str, Any]]
    uncertainty_output: str

    # Visualizer
    visualization_result: dict[str, Any]

    # Reporter
    report_markdown: str

    # RAG / Memory
    rag_analysis_context: str
    rag_stored: bool
    rag_summary: str
    memory_trace: list[MemoryTraceEntry]

    # Observability
    langfuse_trace_id: str
    graph_trace: list[NodeTraceEntry]
    error: str
```

---

## 9. Configuration centralisée

### 9.1 `config/settings.py` — Settings dataclass (frozen)

| Variable env | Défaut | Description |
|-------------|--------|-------------|
| `LLM_PROVIDER` | `ollama` | Provider LLM (ollama / gemini) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL du serveur Ollama |
| `OLLAMA_TEXT_MODEL` | `qwen3:14b` | Modèle pour le texte |
| `OLLAMA_CODE_MODEL` | `qwen2.5-coder:14b` | Modèle pour le code |
| `LLM_TIMEOUT` | `60` | Timeout LLM en secondes |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Modèle Gemini (fallback) |
| `PROFILER_SAMPLE_ROWS` | `5` | Nombre de lignes échantillon |
| `PROFILER_TOP_VALUES` | `10` | Nombre de top values par colonne |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Modèle d'embeddings |
| `CHROMA_DIR` | `./chroma_db` | Répertoire ChromaDB |
| `LANGFUSE_ENABLED` | `false` | Activer le tracing LangFuse |

### 9.2 `config/prompts.yaml` — Tous les prompts

Sections : `profiler`, `analyst`, `categorizer`, `reporter`, `visualizer`, `critic`, `uncertainty`

Chaque section contient un prompt `system` (rôle + règles) et un prompt `human` (template avec variables `{...}`).

---

## 10. Tests et qualité

### 10.1 Suite de tests

| Fichier | Tests | Ce qu'il couvre |
|---------|-------|-----------------|
| `test_data_loader.py` | 9 | CSV, Excel, Parquet, uploads, erreurs |
| `test_profiler_engine.py` | 25 | Shape, types, stats, sérialisation, edge cases |
| `test_analyst.py` | 8 | extract_json, validate, normalize, categorize, format |
| `test_critic.py` | 8 | validate_critique, normalize, formatter, node |
| `test_critic_uncertainty.py` | 32 | Uncertainty scoring complet (rules + LLM mock) |
| `test_reporter.py` | 6 | Chart extraction, insights summary, LLM integration |
| `test_rag.py` | 16 | dataset_id, format/truncate, RAG storage node |
| `test_langfuse.py` | 18 | Monitor lifecycle, spans, callbacks, graceful degradation |
| `test_memory_view.py` | 10 | Memory status, retrieval helpers, constants |
| `test_smoke.py` | 2 | Import validation |
| `test_visualizer_executor.py` | 12 | Chart types, erreurs sandbox, sécurité |
| `test_visualizer_parser.py` | 14 | JSON parsing, validation schema, edge cases |
| **Total** | **180 unit + 7 integration** | — |

### 10.2 Pipeline CI (`make ci`)

```bash
make ci
├── uv run ruff check           # Lint
├── uv run ruff format --check  # Formatage
├── uv run pytest tests/ -v -m "not integration"  # 180 tests
├── uv run pre-commit run --all-files  # Hooks
└── uv run mypy .               # Type checking
```

### 10.3 Outils qualité

- **Ruff** : lint + format (line-length=100, target Python 3.11)
- **mypy** : type checking strict (`check_untyped_defs=true`)
- **pre-commit** : ruff lint, ruff autofix, YAML check, large files, trailing whitespace
- **pytest-cov** : couverture de code

---

## 11. Roadmap des phases

| Phase | Nom | Contenu | Status |
|-------|-----|---------|--------|
| **P0** | Infrastructure | Structure, CI/CD, pre-commit, Docker | ✅ |
| **P1** | Profiler + UI | DataLoader, DataProfiler, ProfilerAgent, Streamlit | ✅ |
| **P2** | Analyst + UI | AnalystAgent, catégorisation, validation JSON | ✅ |
| **P3** | Visualizer + UI | VisualizerAgent, parser, sandbox executor, Plotly | ✅ |
| **P4** | Orchestrator | LangGraph StateGraph, LangFuse, pipeline complet | ✅ |
| **P5** | ChromaDB + RAG | ContextStore, RAGAgent, mémoire entre analyses | ✅ |
| **P6** | Text-to-Code | Agent Q&A, génération de code pandas, sandbox | 🔲 Stub |
| **P7** | Evaluation | LLM-as-judge, scoring qualité | 🔲 Stub |
| **P8** | Critic + Uncertainty | CriticAgent, UncertaintyEstimator, rapport enrichi | ✅ |

---

## 12. Principes de conception

### 12.1 Déterministe d'abord
Le profilage est 100% pandas — aucun LLM. Résultats reproductibles, rapides, et sans hallucination.

### 12.2 Local-first + fallback cloud
Ollama (local) en priorité, Gemini (cloud) en fallback. Pas de dépendance à Internet pour fonctionner.

### 12.3 Outputs structurés
Chaque agent produit du JSON validé par schéma. Fallback markdown si le parsing échoue.

### 12.4 Sandbox sécurisé
L'exécution de code Plotly se fait dans un namespace restreint : pas de `__builtins__`, pas d'imports, uniquement `df`, `pd`, `px`, `go`, `np`.

### 12.5 Dégradation gracieuse
Si un agent échoue, le pipeline continue. Les résultats partiels sont affichés. LangFuse désactivé = aucune erreur.

### 12.6 Configuration par environnement
Aucun paramètre hardcodé. Tout via `.env` + `config/settings.py` (dataclass frozen).

---

## 13. Différenciateurs clés

1. **Architecture multi-agent** avec orchestration LangGraph et streaming progressif
2. **Critique adversariale** — le Critic remet en question les insights de l'Analyst
3. **Quantification de l'incertitude** — score hybride règles + LLM (0-100%)
4. **Rapport enrichi** — le Reporter intègre critique et confiance dans le rapport final
5. **Mémoire RAG** — contexte des analyses précédentes pour enrichir les nouvelles
6. **Sandbox sécurisé** — exécution de code LLM sans risque
7. **Observabilité complète** — LangFuse tracing, diagnostics CPU/RAM/durée par agent
8. **Local-first** — fonctionne sans connexion Internet avec Ollama

---

## 14. Structure du projet

```
autoinsight-ai/
├── agents/                    # Les 6 agents + RAG + mock
│   ├── profiler.py            # ProfilerAgent
│   ├── analyst.py             # AnalystAgent + Categorizer + Formatter
│   ├── critic.py              # CriticAgent + CriticFormatter
│   ├── uncertainty.py         # UncertaintyEstimator + Formatter
│   ├── visualizer.py          # VisualizerAgent
│   ├── reporter.py            # ReporterAgent
│   ├── rag.py                 # RAGAgent
│   └── mock_profiler.py       # Mock data pour tests
│
├── app/                       # Interface Streamlit
│   ├── main.py                # App principale (~1000 lignes)
│   ├── memory_view.py         # Helpers pour le tab Memory
│   └── uncertainty_view.py    # Renderer confidence scores (standalone)
│
├── config/                    # Configuration
│   ├── settings.py            # Settings dataclass (frozen, env-driven)
│   └── prompts.yaml           # Tous les prompts LLM (7 sections)
│
├── orchestration/             # Pipeline LangGraph
│   ├── graph.py               # StateGraph + nodes + streaming
│   └── state.py               # PipelineState TypedDict
│
├── tools/                     # Outils data
│   ├── data_loader.py         # CSV/Excel/Parquet loader
│   ├── profiler_engine.py     # DataProfiler (pandas déterministe)
│   ├── text_to_code.py        # Q&A code gen (stub)
│   └── viz_utils.py           # Visualisation utils (stub)
│
├── visualization/             # Exécution charts
│   ├── schemas.py             # Dataclass contracts
│   ├── parser.py              # Parse LLM JSON → ChartSpec
│   └── executor.py            # Sandbox exec → Plotly Figure
│
├── utils/                     # Utilitaires
│   ├── llm.py                 # LLM client factory (Ollama/Gemini)
│   ├── memory.py              # ChromaDB ContextStore
│   ├── prompt_loader.py       # YAML prompt loader
│   └── langfuse_client.py     # LangFuse singleton monitor
│
├── evaluation/                # Évaluation qualité
│   ├── config.py              # Constantes d'évaluation
│   ├── code_eval.py           # (stub)
│   ├── llm_judge.py           # (stub)
│   └── validators.py          # (stub)
│
├── diagnostics/               # Diagnostics UI
│   └── renderer.py            # Pipeline diagram + métriques + traces
│
├── tests/                     # 187 tests (180 unit + 7 integration)
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_profiler_engine.py
│   ├── test_analyst.py
│   ├── test_critic.py
│   ├── test_critic_uncertainty.py
│   ├── test_reporter.py
│   ├── test_rag.py
│   ├── test_langfuse.py
│   ├── test_memory_view.py
│   ├── test_smoke.py
│   ├── test_visualizer_executor.py
│   └── test_visualizer_parser.py
│
├── data/                      # Données exemple
│   └── sample_sales.csv       # 60 lignes, e-commerce
│
├── docs/                      # Documentation
│   ├── TASK-P1.md             # Phase 1: Profiler
│   ├── TASK-P2.md             # Phase 2: Analyst
│   ├── TASK-P3.md             # Phase 3: Visualizer
│   ├── critic_agent_design.md # Design doc Critic
│   └── phase8_design.md       # Design doc Phase 8
│
├── pyproject.toml             # Config projet + dépendances
├── Makefile                   # Commandes dev (lint, test, run, ci)
├── Project_Spec.md            # Spécification technique complète
├── TASKS.md                   # Breakdown complet des tâches
└── README.md                  # Quick start
```
