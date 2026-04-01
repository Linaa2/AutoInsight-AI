# AutoInsight AI

Système multi-agent d'analyse de données automatique — LangGraph + Ollama, avec Gemini en option.

## Stack
| Composant | Outil |
|---|---|
| LLM primaire | Ollama (`qwen3:4b` light / `qwen3:14b` text / `qwen2.5-coder:14b` code par défaut) |
| LLM optionnel | Gemini (`gemini-1.5-flash`) |
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
