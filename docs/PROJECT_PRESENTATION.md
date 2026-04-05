# AutoInsight AI — Documentation Projet

> **Système multi-agent d'analyse de données automatique**
> Exploitant les technologies d'IA Générative (LLM) avec une architecture agentic avancée

---

## 1. Présentation du projet

**AutoInsight AI** est une application innovante qui transforme un simple fichier de données (CSV, Excel) en un **rapport d'analyse exécutif complet** — automatiquement, sans intervention humaine.

Le système repose sur une **architecture multi-agent** où 6 agents LLM spécialisés collaborent séquentiellement via un graphe d'orchestration. Chaque agent a un rôle précis et passe le relais au suivant, en enrichissant un état partagé.

**Ce qui rend le projet unique** :
- Ce n'est **pas** un chatbot ni un RAG basique — c'est un pipeline agentic complet
- Les agents se **critiquent mutuellement** (Critic Agent) et **quantifient leur incertitude** (Uncertainty Estimator)
- Le profilage des données est **100% déterministe** (pandas) — pas d'hallucination sur les chiffres
- L'exécution de code généré par LLM se fait dans un **sandbox sécurisé**
- Le système fonctionne **entièrement en local** (Ollama) avec fallback cloud (Gemini)

---

## 2. Stack technique et justification des choix

| Composant | Technologie | Pourquoi ce choix |
|-----------|-------------|-------------------|
| **Orchestration** | **LangGraph** (StateGraph) | Graphe d'agents avec état partagé typé, streaming natif, contrôle fin du flux |
| **LLM local** | **Ollama** (Mistral 7B, Qwen3 14B) | Local-first, gratuit, pas de dépendance réseau, confidentialité des données |
| **LLM code** | **Ollama** (Qwen2.5-coder 14B) | Modèle spécialisé code pour la génération de charts Plotly |
| **LLM cloud** | **Google Gemini 1.5 Flash** | Fallback gratuit si Ollama indisponible |
| **Observabilité** | **LangFuse** (Docker self-hosted) | Tracing complet des appels LLM, métriques, debugging |
| **Mémoire** | **ChromaDB** + nomic-embed-text | Base vectorielle locale pour RAG entre analyses |
| **UI** | **Streamlit** | Prototypage rapide, rendu progressif, widgets interactifs |
| **Visualisation** | **Plotly** Express + Graph Objects | Charts interactifs exportables |
| **Data** | **Pandas** + NumPy | Profilage déterministe fiable |
| **Qualité** | **Ruff** + **mypy** + **pytest** + **pre-commit** | Pipeline CI complète, code typé, 180+ tests |
| **Packaging** | **uv** + hatchling | Gestionnaire de dépendances rapide et reproductible |
| **Infra** | **Docker Compose** | LangFuse self-hosted (PostgreSQL + LangFuse server) |

---

## 3. LangGraph — Orchestration multi-agent

### Qu'est-ce que LangGraph ?

LangGraph est un framework de LangChain pour construire des **workflows multi-agents** sous forme de **graphes d'états**. Contrairement à un simple enchaînement de fonctions, LangGraph offre :
- Un **état typé partagé** (`PipelineState` — TypedDict) entre tous les nœuds
- Le **streaming** : chaque nœud émet son résultat dès qu'il termine
- La **traçabilité** : chaque nœud enregistre son statut, sa durée, ses erreurs
- La possibilité de **branching conditionnel** (non utilisé ici, mais extensible)

### Comment on l'utilise dans AutoInsight AI

```
START → profiler → analyst → critic → uncertainty → visualizer → reporter → rag_storage → END
```

Chaque nœud est une fonction Python qui :
1. **Lit** ses inputs depuis le `PipelineState` partagé
2. **Délègue** à la classe agent correspondante dans `agents/`
3. **Écrit** ses outputs + une entrée de trace dans le state
4. **Capture** la télémétrie (CPU, RAM, durée, modèle utilisé, tokens)

Le **streaming** permet au frontend Streamlit d'afficher les résultats **progressivement** : le tab "Profile" apparaît pendant que l'Analyst travaille encore.

### Fichiers impliqués

| Fichier | Rôle |
|---------|------|
| `orchestration/graph.py` | Définition du graphe, fonctions nœuds, `build_graph()`, `run_analysis()`, `stream_analysis()` |
| `orchestration/state.py` | `PipelineState` (TypedDict) — contrat de données entre agents |

### PipelineState — les données qui circulent

```python
class PipelineState(TypedDict, total=False):
    df_dict: list[dict]           # DataFrame sérialisé (input)
    file_name: str                # Nom du fichier uploadé
    dataset_id: str               # ID unique du dataset (pour RAG)

    profile_data: dict            # Profil JSON (sortie Profiler)
    profile_markdown: str         # Description narrative (sortie Profiler)

    insights: list[dict]          # Insights structurés (sortie Analyst)
    insights_markdown: str        # Insights en markdown (sortie Analyst)

    critiques: list[dict]         # Critiques par insight (sortie Critic)
    critic_output: str            # Critic review en markdown

    confidence_scores: list[dict] # Scores 0-100% par insight (sortie Uncertainty)
    uncertainty_output: str       # Tableau de confiance en markdown

    visualization_result: dict    # Charts Plotly (sortie Visualizer)
    report_markdown: str          # Rapport exécutif final (sortie Reporter)

    graph_trace: list[dict]       # Trace d'exécution par nœud (diagnostics)
    langfuse_trace_id: str        # ID trace LangFuse
```

---

## 4. LangFuse — Observabilité LLM

### Qu'est-ce que LangFuse ?

LangFuse est une plateforme d'**observabilité pour les applications LLM** (LLMOps). Elle permet de tracer, debugger et évaluer les appels aux modèles de langage.

### Déploiement

LangFuse est **self-hosted** via Docker Compose :
- **PostgreSQL** (port 5433) pour le stockage des traces
- **LangFuse Server** (port 3001) pour l'UI et l'API
- **Setup automatisé** : `scripts/setup_langfuse.py` génère les credentials et seed le projet automatiquement (pas de création de compte manuelle)

```bash
make langfuse-up    # Démarre LangFuse
make langfuse-down  # Arrête LangFuse
```

### Ce qu'on trace dans notre projet

| Niveau | Ce qui est capturé |
|--------|--------------------|
| **Run** | 1 trace par analyse complète (begin_run → end_run) |
| **Agent** | 1 span par agent (profiler, analyst, critic, etc.) avec métadonnées |
| **LLM** | Chaque appel LLM via callbacks LangChain (prompt, réponse, tokens, durée) |
| **RAG** | Événements de stockage/retrieval ChromaDB |

### Architecture dans le code

```python
# Singleton — importé partout
from utils.langfuse_client import monitor

# Au début d'une analyse
trace_id = monitor.begin_run(dataset_id="sales.csv", run_id="abc-123")

# Dans chaque nœud du graphe
callbacks = monitor.get_llm_callbacks()  # Callbacks pour le LLM
with monitor.node_span("profiler", metadata={"rows": 1000}):
    result = agent.describe(profile, callbacks=callbacks)

# Événement RAG
monitor.log_event("rag-store-profile", output={"chunks": 5})

# En fin d'analyse
monitor.end_run(status="success")
```

### Dégradation gracieuse

**Point clé** : si LangFuse est désactivé (`LANGFUSE_ENABLED=false`) ou inaccessible, **aucune erreur ne se produit**. Toutes les méthodes retournent des valeurs vides et un `_NoOpSpan` (null-object pattern). L'application fonctionne normalement sans observabilité.

### Fichiers impliqués

| Fichier | Rôle |
|---------|------|
| `utils/langfuse_client.py` | `LangFuseMonitor` singleton + `_NoOpSpan` + `is_langfuse_enabled()` |
| `docker-compose.langfuse.yml` | PostgreSQL + LangFuse Server |
| `scripts/setup_langfuse.py` | Génération automatique des credentials |

---

## 5. Les 6 agents du pipeline

### 5.1 📊 Profiler — Profilage déterministe + interprétation LLM

| | |
|---|---|
| **Fichiers** | `tools/profiler_engine.py` (déterministe) + `agents/profiler.py` (LLM) |
| **Modèle** | Texte (Mistral 7B / Qwen3 14B) |

- `DataProfiler` calcule **sans LLM** : shape, types, stats (mean, median, std, skew, kurtosis), missing values, duplicates, top values, échantillons
- `ProfilerAgent` envoie le profil JSON au LLM qui génère une **description narrative markdown**
- **Avantage** : les chiffres sont toujours corrects (pandas), le LLM ne fait que les interpréter

### 5.2 💡 Analyst — Génération d'insights structurés

| | |
|---|---|
| **Fichier** | `agents/analyst.py` |
| **Modèle** | Texte |

- Génère **3-6 insights** structurés à partir du profil
- Chaque insight : titre, observation, hypothèse, recommandation, priorité, catégorie
- **Validation robuste** : `extract_json()` pour parser le JSON bruité du LLM, `validate_insight()` pour vérifier les champs, `fallback_parse_markdown()` si le JSON échoue
- **Catégorisation** par mots-clés (rapide) ou par LLM (précise)

### 5.3 🔎 Critic — Revue adversariale des insights

| | |
|---|---|
| **Fichier** | `agents/critic.py` |
| **Modèle** | Texte — 1 appel LLM par insight |

- **Adversarial review** : questionne chaque insight de l'Analyst
- Identifie forces, faiblesses, hypothèses alternatives
- Attribue un **verdict** (supported / partially_supported / weak) et un niveau de confiance
- Démontre la **collaboration inter-agents** : l'Analyst propose, le Critic challenge

### 5.4 🎯 Uncertainty Estimator — Score de confiance hybride

| | |
|---|---|
| **Fichier** | `agents/uncertainty.py` |
| **Méthode** | Règles déterministes (50%) + LLM (50%) |

- Score **0-100%** pour chaque insight sur 4 axes :
  - 📦 **Data Quality** (règles) : taille du dataset, missing values, duplicates
  - 🔬 **Specificity** (règles) : colonnes mentionnées, chiffres cités
  - 📐 **Statistical Evidence** (LLM) : solidité des observations
  - 🧐 **Critic Assessment** (LLM) : sévérité des critiques
- **Approche hybride** : 2 drivers déterministes + 2 drivers LLM = résultat fiable et explicable

### 5.5 📈 Visualizer — Génération automatique de charts

| | |
|---|---|
| **Fichiers** | `agents/visualizer.py` + `visualization/parser.py` + `visualization/executor.py` |
| **Modèle** | Code (Qwen2.5-coder 14B) — modèle spécialisé |

- Le LLM génère des **spécifications de charts** en JSON (titre, type, code Plotly)
- Le **parser** valide le JSON et extrait les `ChartSpec`
- L'**executor** exécute le code dans un **sandbox sécurisé** (pas de `__builtins__`, pas d'imports)
- Types : bar, scatter, histogram, line, box, heatmap, pie, treemap, etc.

### 5.6 📄 Reporter — Rapport exécutif final

| | |
|---|---|
| **Fichier** | `agents/reporter.py` |
| **Modèle** | Texte |

- Synthétise **tous** les outputs en un rapport markdown professionnel
- **Intègre dynamiquement** le Critic Review et les Confidence Scores dans le prompt
- Sections : Executive Summary → Dataset Description → Key Insights → Visualizations → Recommendations → Limitations
- Le LLM qualifie les insights low/medium confidence dans le rapport

---

## 6. RAG — Mémoire entre analyses (ChromaDB)

| | |
|---|---|
| **Fichiers** | `agents/rag.py` + `utils/memory.py` |
| **Base vectorielle** | ChromaDB (local, persisté) |
| **Embeddings** | Ollama nomic-embed-text |

**Comment ça marche** :
1. À la fin de chaque analyse, `rag_storage_node` stocke profil + insights + rapport dans ChromaDB
2. Chaque dataset a un `dataset_id` unique → **isolation** entre datasets
3. Lors d'une nouvelle analyse du même dataset, le Reporter reçoit le **contexte des analyses précédentes**
4. Permet de **détecter des évolutions** dans les données au fil du temps

---

## 7. Application Streamlit

### 7.1 Flux utilisateur

1. **Upload** d'un fichier CSV ou Excel via la sidebar
2. **Lancement** de l'analyse (bouton)
3. **Rendu progressif** — les tabs apparaissent au fur et à mesure que les agents terminent :
   - ✓ Agent terminé
   - ⏳ Agent en cours
   - "Waiting…" Agent en attente
4. **Exploration** des résultats dans les tabs interactifs
5. **Téléchargement** des rapports (markdown, JSON)

### 7.2 Les 7 tabs de l'interface

| Tab | Contenu |
|-----|---------|
| 📊 **Profile** | KPIs (lignes, colonnes, duplicates, % missing, mémoire) + détail par colonne + échantillon |
| 💡 **Insights** | Résumé par priorité et catégorie + cards interactives par insight (observation, hypothèse, recommandation) |
| 🔎 **Evaluation** | **Critic Review** (verdicts, forces/faiblesses par insight) + **Confidence Scores** (progress bars, 4 drivers) — tab unifié |
| 📈 **Visualizations** | Charts Plotly interactifs + code source + explication par chart |
| 📄 **Report** | Rapport exécutif markdown complet + bouton de téléchargement |
| 🧠 **Memory** | Historique RAG, insights prioritaires, insights par catégorie |
| 🔧 **Diagnostics** | Pipeline diagram interactif (React Flow), métriques CPU/RAM/durée par agent, trace JSON téléchargeable |

### 7.3 Features notable

- **Pipeline diagram** (React Flow) : visualisation interactive du graphe d'agents avec statut par nœud
- **Télémétrie par agent** : CPU %, RAM (RSS), durée, modèle utilisé, tokens (quand disponible)
- **Rendu progressif** : streaming LangGraph → chaque tab se remplit en temps réel
- **Téléchargement** : insights (.md, .json), rapport (.md), critic review (.md), trace diagnostics (.json)

### Fichier principal

| Fichier | Rôle |
|---------|------|
| `app/main.py` (~1 000 lignes) | Application Streamlit complète — upload, streaming, rendering, téléchargement |
| `app/memory_view.py` | Helpers pour le tab Memory (retrieval ChromaDB) |
| `diagnostics/renderer.py` | Rendu du tab Diagnostics (React Flow + métriques + traces) |
| `diagnostics/telemetry.py` | Collecte CPU/RAM/GPU via psutil |

---

## 8. Bonnes pratiques de développement

### 8.1 Versionnement Git

**Workflow Git Feature Branch** :
- Branche `main` : production stable
- Branche `develop` : intégration
- Branches `feature/*` : une branche par fonctionnalité/phase

**Branches du projet** :

| Branche | Phase/Fonctionnalité |
|---------|---------------------|
| `feature/p1-profiler` | Phase 1 — Profiler + DataLoader |
| `feature/P4-reporter` | Phase 4 — Reporter + Orchestration |
| `feature/visualizer` | Phase 3 — Visualizer |
| `feat/llm-router` | Routing multi-modèles (texte vs code) |
| `feature/rag-agent` | Phase 5 — RAG + ChromaDB |
| `feature/logger` | Logging structuré |
| `fix/langfuse` | Fix autorisation LangFuse |
| `feature/P7-Evaluation` | Phase 7 — Evaluation |
| `feature/P8-uncertainty-estimator` | Phase 8 — Uncertainty Estimator |
| `feature/critic-agent` | Phase 8 — Critic Agent + intégration |

**33 commits** au total, merges propres entre branches.

### 8.2 Pipeline CI locale (`make ci`)

```
make ci
├── ruff check              → Lint (100+ règles activées)
├── ruff format --check     → Vérification du formatage
├── pytest tests/ -v        → 180 tests unitaires
├── pre-commit run --all    → Hooks de qualité
└── mypy .                  → Type checking strict (59 fichiers)
```

### 8.3 Pre-commit hooks

Configuration `.pre-commit-config.yaml` — exécuté automatiquement à chaque commit :

| Hook | Ce qu'il fait |
|------|---------------|
| **Ruff Lint** | Détecte les erreurs de style et bugs potentiels |
| **Ruff Auto-Fix** | Corrige automatiquement les problèmes détectables |
| **check-yaml** | Valide la syntaxe des fichiers YAML |
| **check-added-large-files** | Bloque les fichiers > 1 Mo |
| **end-of-file-fixer** | Assure un newline en fin de fichier |
| **trailing-whitespace** | Supprime les espaces en fin de ligne |

### 8.4 Ruff — Lint et formatage

- **Line length** : 100 caractères
- **Target** : Python 3.11
- **Règles activées** : E, W, F (pyflakes), I (isort), B (bugbear), C4 (comprehensions), UP (pyupgrade), ARG (unused args), SIM (simplify), TCH (type-checking), PTH (pathlib), RUF (ruff-specific)
- **Format** : double quotes, cohérent sur 59 fichiers

### 8.5 mypy — Type checking

- Type checking strict avec `check_untyped_defs = true`
- **59 fichiers** typés et vérifiés
- `PipelineState` est un `TypedDict` — contrat de données typé entre agents
- `Settings` est un `dataclass(frozen=True)` — configuration immutable

### 8.6 Tests

| Type | Nombre | Ce qu'ils testent |
|------|--------|-------------------|
| **Unit tests** | 180 | Logique pure sans LLM — parsing, validation, formatting, profiling, sandbox |
| **Integration tests** | 7 | Appels LLM réels — marqués `@pytest.mark.integration`, exclus du CI rapide |
| **Smoke tests** | 2 | Import validation — vérifier que le package est installable |
| **Total** | 187 | — |

Les tests unitaires s'exécutent en **~4 secondes** sans aucune dépendance externe.

### 8.7 Configuration centralisée

- **`config/settings.py`** : `Settings` dataclass frozen — toutes les variables d'environnement centralisées, immutables à runtime
- **`config/prompts.yaml`** : tous les prompts LLM (7 sections) — modifiables sans toucher au code
- **`.env`** : valeurs spécifiques à l'environnement (jamais commité)
- **Pas de hardcoding** : tout est configurable via environnement

### 8.8 Makefile

Commandes standardisées pour toute l'équipe :

| Commande | Action |
|----------|--------|
| `make install` | Installe dépendances + pre-commit hooks |
| `make lint` | Ruff check |
| `make format` | Vérifie le formatage |
| `make test` | 180 tests unitaires |
| `make test-all` | Tous les tests (incluant LLM) |
| `make ci` | Pipeline CI complète (lint + format + tests + pre-commit + mypy) |
| `make run` | Lance l'app Streamlit |
| `make langfuse-up` | Démarre LangFuse (Docker) |
| `make langfuse-down` | Arrête LangFuse |
| `make fix` | Auto-fix formatage + whitespace |

---

## 9. Sécurité

### Sandbox d'exécution de code

Le Visualizer génère du code Plotly via LLM. Ce code est exécuté dans un **namespace restreint** :

```python
namespace = {
    "__builtins__": {},   # Aucun builtin → pas d'import, pas d'eval, pas d'exec
    "df": df,             # Seul le DataFrame est accessible
    "pd": pandas,
    "px": plotly.express,
    "go": plotly.graph_objects,
    "np": numpy,
}
exec(code, namespace)
```

**Pas d'accès** au filesystem, au réseau, aux variables d'environnement, ni à aucune fonction Python standard.

### Validation des outputs LLM

- `extract_json()` : extraction sûre de JSON depuis une réponse LLM bruitée
- `validate_insight()` : vérification de tous les champs requis
- `validate_critique()` : idem pour les critiques
- Fallback markdown si le parsing JSON échoue — jamais de crash

---

## 10. Structure du repo — Fichiers importants

### Agents (cœur du système)

| Fichier | Rôle | LoC |
|---------|------|-----|
| `agents/profiler.py` | Interprétation LLM du profil → markdown narratif | ~80 |
| `agents/analyst.py` | Génération d'insights structurés + catégorisation + validation | ~350 |
| `agents/critic.py` | Revue adversariale — critique chaque insight | ~250 |
| `agents/uncertainty.py` | Score de confiance hybride (règles + LLM) | ~300 |
| `agents/visualizer.py` | Génération de charts via LLM code | ~200 |
| `agents/reporter.py` | Rapport exécutif intégrant tous les outputs | ~230 |
| `agents/rag.py` | Stockage/retrieval ChromaDB | ~150 |

### Orchestration

| Fichier | Rôle |
|---------|------|
| `orchestration/graph.py` | Graphe LangGraph — nœuds, edges, streaming, télémétrie |
| `orchestration/state.py` | `PipelineState` TypedDict — contrat de données typé |

### Outils

| Fichier | Rôle |
|---------|------|
| `tools/data_loader.py` | Chargement CSV/Excel/Parquet — détection auto du format |
| `tools/profiler_engine.py` | `DataProfiler` — profilage déterministe pandas (25 métriques) |

### Visualisation

| Fichier | Rôle |
|---------|------|
| `visualization/schemas.py` | Dataclass contracts (ChartSpec, VisualizerRequest, etc.) |
| `visualization/parser.py` | Parse JSON LLM → ChartSpec validés |
| `visualization/executor.py` | Sandbox exec → Plotly Figure |

### UI

| Fichier | Rôle |
|---------|------|
| `app/main.py` | Application Streamlit (~1 000 lignes) — upload, streaming, 7 tabs |
| `diagnostics/renderer.py` | Tab Diagnostics — pipeline diagram (React Flow) + métriques |
| `diagnostics/telemetry.py` | Collecte CPU/RAM/GPU via psutil |

### Utilitaires

| Fichier | Rôle |
|---------|------|
| `utils/llm.py` | Factory LLM (Ollama/Gemini) + routing texte/code |
| `utils/memory.py` | `ContextStore` ChromaDB — stockage et retrieval vectoriel |
| `utils/prompt_loader.py` | Chargeur YAML partagé pour les prompts |
| `utils/langfuse_client.py` | `LangFuseMonitor` singleton + dégradation gracieuse |

### Configuration

| Fichier | Rôle |
|---------|------|
| `config/settings.py` | `Settings` dataclass frozen — 15+ variables d'environnement |
| `config/prompts.yaml` | 7 sections de prompts LLM (system + human) |
| `.pre-commit-config.yaml` | 6 hooks de qualité (ruff, yaml, whitespace, etc.) |
| `docker-compose.langfuse.yml` | LangFuse self-hosted (PostgreSQL + LangFuse Server) |
| `Makefile` | 12 commandes standardisées (ci, test, run, langfuse, etc.) |
| `pyproject.toml` | Dépendances, config ruff/mypy/pytest |

### Tests

| Fichier | Tests | Cible |
|---------|-------|-------|
| `tests/test_profiler_engine.py` | 25 | DataProfiler — shape, types, stats, edge cases |
| `tests/test_critic_uncertainty.py` | 32 | Uncertainty scoring complet (rules + LLM mock) |
| `tests/test_langfuse.py` | 18 | Monitor lifecycle, spans, callbacks |
| `tests/test_rag.py` | 16 | RAG storage, dataset_id, format/truncate |
| `tests/test_visualizer_parser.py` | 14 | JSON parsing, chart schema validation |
| `tests/test_visualizer_executor.py` | 12 | Sandbox exec, security, chart types |
| `tests/test_memory_view.py` | 10 | Memory status, retrieval helpers |
| `tests/test_data_loader.py` | 9 | CSV/Excel/Parquet, uploads, erreurs |
| `tests/test_analyst.py` | 8 | extract_json, validate, categorize |
| `tests/test_critic.py` | 8 | validate_critique, normalize, formatter |
| `tests/test_reporter.py` | 6 | Chart extraction, insights summary |
| `tests/test_smoke.py` | 2 | Import validation |

---

## 11. Résumé des objectifs réalisés

| Critère d'évaluation | Ce qu'on a fait |
|----------------------|-----------------|
| **Qualité du code** | Ruff lint, mypy strict, 180 tests, pre-commit hooks, pipeline CI (`make ci`), code modulaire |
| **Bonne utilisation de l'IA Gen** | 6 agents LLM spécialisés, routing texte/code, prompts YAML configurables, validation JSON robuste |
| **Originalité et technicité** | Critique adversariale, quantification d'incertitude hybride, sandbox sécurisé, mémoire RAG inter-analyses |
| **Architecture cohérente** | LangGraph StateGraph typé, PipelineState contrat, agents découplés, configuration centralisée |
| **Collaboration Git** | Feature branches, 33 commits, merges propres, 3 développeurs |
| **Observabilité** | LangFuse self-hosted (Docker), tracing par run/agent/LLM, télémétrie CPU/RAM |
| **UI fonctionnelle** | Streamlit avec rendu progressif, 7 tabs interactifs, diagnostics pipeline, téléchargements |
