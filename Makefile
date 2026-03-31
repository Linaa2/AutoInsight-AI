.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dev dependencies and pre-commit hooks
	uv sync --extra dev
	uv run pre-commit install

pre-commit: ## Run all pre-commit checks
	uv run pre-commit run --all-files

lint: ## Run ruff linting
	uv run ruff check

typecheck: ## Run mypy type checking
	uv run mypy .

format: ## Check code formatting with ruff
	uv run ruff format --check

fix: ## Auto-fix formatting (trailing whitespace, end-of-file)
	uv run ruff format
	uv run pre-commit run trailing-whitespace --all-files || true
	uv run pre-commit run end-of-file-fixer --all-files || true

test: ## Run pytest (unit tests only — fast)
	uv run pytest tests/ -v -m "not integration"

test-all: ## Run all tests including LLM integration tests
	uv run pytest tests/ -v

check: lint format test pre-commit ## Run all checks (linting, formatting, tests, pre-commit hooks)

ci: check typecheck ## Run all checks including type checking (for CI pipelines)

run: ## Run the Streamlit app
	uv run streamlit run app/main.py

langfuse-up: ## Start LangFuse — auto-generates credentials on first run
	uv run python scripts/setup_langfuse.py
	docker compose -f docker-compose.langfuse.yml up -d
	uv run python scripts/seed_langfuse_membership.py
	@echo ""
	@echo "  LangFuse UI → http://localhost:3001"
	@echo "  Credentials → .env.langfuse (admin: admin@autoinsight.local)"
	@echo ""

langfuse-down: ## Stop LangFuse (data preserved)
	docker compose -f docker-compose.langfuse.yml down

langfuse-reset: ## Destroy LangFuse data and rotate all credentials
	@echo "⚠  This will destroy all LangFuse traces and rotate credentials."
	@read -p "   Continue? [y/N] " _c && [ "$$_c" = y ] || exit 1
	docker compose -f docker-compose.langfuse.yml down -v
	rm -f .env.langfuse
	uv run python scripts/setup_langfuse.py
	docker compose -f docker-compose.langfuse.yml up -d
	uv run python scripts/seed_langfuse_membership.py

.PHONY: help install pre-commit lint format fix test check run langfuse-up langfuse-down langfuse-reset
