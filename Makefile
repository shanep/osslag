VENV ?= .venv

.PHONY: bootstrap install lint fmt typecheck test check rust-check build clean help

help:
	@echo "Makefile commands:"
	@echo "  bootstrap   - Check uv installation and sync dependencies"
	@echo "  install     - Install the package in editable mode"
	@echo "  lint        - Lint the code using ruff"
	@echo "  fmt         - Format the code using ruff"
	@echo "  typecheck   - Type check the code using pyright"
	@echo "  test        - Run tests using pytest"
	@echo "  check       - Run every check CI and the pre-push hook run"
	@echo "  clean       - Clean up build artifacts and caches"

bootstrap:
	@echo "check the uv installation..."
	@if ! command -v uv >/dev/null 2>&1; then \
	  echo "uv not found."; \
	  echo "Check https://github.com/astral-sh/uv for installation instructions."; \
	  exit 1; \
	else \
	  echo "uv is installed: $$(uv --version)"; \
	fi
	@echo "Syncing dependencies using uv..."
	@uv sync

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run pyright src tests

test:
	uv run pytest

check: lint typecheck test rust-check

rust-check:
	cargo fmt --check
	cargo clippy --all-targets --locked -- -D warnings
	cargo test --locked

install:
	uv tool install . -e

build:
	uv build

clean:
	uv cache clean
	rm -rf $(VENV)
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".uv" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
