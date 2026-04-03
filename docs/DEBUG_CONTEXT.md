# Problème : Pipeline LLM crash au Profiler Agent

## Contexte

AutoInsight AI est un système multi-agent (LangGraph) d'analyse de données avec 6 agents LLM. Le problème est que le **Profiler Agent** (premier agent du pipeline) crash systématiquement au moment de l'appel LLM.

Le projet a récemment subi un merge depuis `develop` qui a introduit un nouveau tier de modèle (`OLLAMA_LIGHT_MODEL`) et un module de contexte optimisé (`agents/profiler_context.py`).

## Environnement

- **OS** : Ubuntu (WSL2), 7.6 GB RAM total
- **Python** : 3.12.3
- **Ollama** : modèles installés = `mistral:latest` (4.4 GB), `qwen3:4b` (2.5 GB), `nomic-embed-text` (274 MB)
- **Gemini** : clé API Google active (`GOOGLE_API_KEY` configurée)
- **LangChain** : `langchain-ollama`, `langchain-google-genai`

## Configuration actuelle (.env)

```env
LLM_PROVIDER=ollama
GOOGLE_API_KEY=AIzaSyA4dm...
GEMINI_MODEL=gemini-1.5-flash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEXT_MODEL=mistral
OLLAMA_CODE_MODEL=mistral
OLLAMA_LIGHT_MODEL=mistral
OLLAMA_EMBED_MODEL=nomic-embed-text
LLM_TIMEOUT=120
```

## Ce qu'on a testé

### 1. Avec Ollama (`LLM_PROVIDER=ollama`)

**Test direct curl — fonctionne :**
```bash
curl http://localhost:11434/api/generate -d '{"model":"mistral","prompt":"say hello","stream":false}'
# → Répond correctement
```

**Test Python simple — fonctionne :**
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
llm.invoke("Say hello")
# → Répond correctement
```

**Test via le pipeline (Profiler Agent) — CRASH / TIMEOUT :**
```
Profiler failed: model requires more system memory (4.5 GiB) than is available
# OU
Profiler failed: timeout after 120 seconds
```

Le modèle répond à des prompts courts mais **timeout sur le vrai prompt du profiler** (qui inclut un JSON de profil de dataset).

### 2. Avec Gemini (`LLM_PROVIDER=gemini`)

**Problème 1** : `gemini-1.5-flash` retourne 404 (modèle retiré de l'API)

**Problème 2** : Quand on change pour `gemini-2.0-flash`, on obtient 429 (rate limit) puis timeout.

## Architecture du LLM routing

Le fichier `utils/llm.py` définit 3 tiers de modèles :

```python
_TASK_TO_KIND = {
    "profiler": "light",      # → OLLAMA_LIGHT_MODEL (mistral)
    "analyst": "text",        # → OLLAMA_TEXT_MODEL (mistral)
    "critic": "text",
    "reporter": "text",
    "uncertainty": "light",
    "categorizer": "light",
    "visualizer": "code",     # → OLLAMA_CODE_MODEL (mistral)
}
```

Quand `LLM_PROVIDER=gemini`, tous les tiers utilisent `GEMINI_MODEL` :
```python
def _default_model(kind):
    if settings.LLM_PROVIDER == "gemini":
        return settings.GEMINI_MODEL   # Tous les tiers → même modèle Gemini
    if kind == "light":
        return settings.OLLAMA_LIGHT_MODEL
    ...
```

La factory LLM (`_build`) instancie soit `ChatOllama` soit `ChatGoogleGenerativeAI` :
```python
def _build(model, *, timeout=None, keep_alive=None):
    if settings.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=model, convert_system_message_to_human=True)
    return ChatOllama(model=model, base_url=settings.OLLAMA_BASE_URL)
```

## Le Profiler Agent

`agents/profiler.py` utilise maintenant un module `agents/profiler_context.py` qui construit un **contexte compact** (au lieu d'envoyer le profil JSON complet). Le prompt utilise la variable `{profile_summary_json}`.

```python
class ProfilerAgent:
    def describe(self, profile, callbacks=None, detail_mode=None):
        ctx = build_profiler_prompt_context(profile, mode)  # contexte compact
        profile_summary_json = json.dumps(ctx, indent=2, default=str)
        llm = self._llm_client.get_task_llm("profiler")  # → "light" tier
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._prompts["system"]),
            ("human", self._prompts["human"]),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"profile_summary_json": profile_summary_json})
        return result
```

Les prompts sont dans `config/prompts.yaml` section `profiler:` — le human prompt utilise `{profile_summary_json}`.

## Ce qu'il faut résoudre

1. **Avec Ollama** : le modèle `mistral` répond à des prompts courts mais **timeout/crash sur le prompt complet du profiler**. Est-ce un problème de taille de contexte ? De RAM insuffisante pendant l'inférence ? De timeout trop court ?

2. **Avec Gemini** : `gemini-1.5-flash` n'existe plus (404). Quel est le bon nom de modèle à utiliser avec `langchain-google-genai` ? Est-ce que `gemini-2.0-flash` ou `gemini-2.5-flash-preview-04-17` fonctionne ?

3. **RAM** : 7.6 GB total, VS Code Server consomme ~1.5 GB, Ollama + modèle ~3.5 GB. Est-ce suffisant pour faire de l'inférence avec mistral 7B sur un prompt de ~2000 tokens ?

## Fichiers pertinents

- `utils/llm.py` — Factory LLM, routing par tâche
- `config/settings.py` — Settings dataclass (toutes les variables env)
- `agents/profiler.py` — Profiler Agent
- `agents/profiler_context.py` — Construction du contexte compact
- `config/prompts.yaml` — Prompts (section `profiler:`)
- `orchestration/graph.py` — Pipeline LangGraph (profiler_node)
- `.env` — Configuration runtime
