# AutoInsight AI

Système multi-agent d'analyse de données automatique — LangGraph + Gemini + Ollama.

## Stack
| Composant | Outil |
|---|---|
| LLM primaire | Gemini 1.5 Flash |
| LLM fallback | Ollama + Mistral 7B |
| Orchestration | LangGraph |
| Monitoring | LangFuse |
| Mémoire | ChromaDB |
| UI | Streamlit |

## Installation
```bash
git clone https://github.com/TON_USERNAME/autoinsight-ai.git
cd autoinsight-ai
# Créer l'environnement virtuel
uv venv .venv

# L'activer
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Installer le projet + dépendances dev
uv pip install -e ".[dev]"

# Copier le fichier d'environnement
cp .env.example .env
# → Remplir .env avec tes clés

# Lancer l'app
streamlit run app/main.py
```