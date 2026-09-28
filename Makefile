.DEFAULT_GOAL := help
.PHONY: help install dev test lint format typecheck demo graph docker-build docker-up clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies (incl. dev tools and the postgres extra)
	uv sync --all-extras

dev:  ## Run API + operator UI with auto-reload on http://localhost:8000
	uv run uvicorn support_agent.api.app:create_app --factory --reload --port 8000

test:  ## Run the test-suite (no API keys needed)
	uv run pytest

lint:  ## Ruff lint + format check + mypy (strict)
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:  ## Auto-format and fix lint issues
	uv run ruff format src tests
	uv run ruff check --fix src tests

demo:  ## Run all demo tickets through the agent and print a summary
	uv run python -m support_agent.cli demo

graph:  ## Print the LangGraph graph as Mermaid
	uv run python -m support_agent.cli graph

docker-build:  ## Build the Docker image
	docker compose build

docker-up:  ## Start the app in Docker (fake LLM by default)
	docker compose up --build

clean:  ## Remove local databases and caches
	rm -rf data .pytest_cache .mypy_cache .ruff_cache
