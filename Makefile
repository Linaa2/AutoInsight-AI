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

format: ## Check code formatting with ruff
	uv run ruff format --check

fix: ## Auto-fix formatting (trailing whitespace, end-of-file)
	uv run ruff format
	uv run pre-commit run trailing-whitespace --all-files || true
	uv run pre-commit run end-of-file-fixer --all-files || true

test: ## Run pytest
	uv run pytest tests/ -v

check: lint format test pre-commit ## Run all checks (linting, formatting, tests, pre-commit hooks)

run: ## Run the Streamlit app
	uv run streamlit run app/main.py

.PHONY: help install pre-commit lint format fix test check run
