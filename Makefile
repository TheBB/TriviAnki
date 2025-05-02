# Convenience targets

.PHONY: sync
sync:
	uv sync --all-packages --group dev


# Linting targets

.PHONY: ruff-format
ruff-format:
	uv run ruff format

.PHONY: ruff-check
ruff-check:
	uv run ruff check --fix

.PHONY: ruff
ruff: ruff-format ruff-check


# Test targets

.PHONY: mypy
mypy:
	uv run mypy

.PHONY: ruff-verify
ruff-verify:
	uv run ruff check
	uv run ruff format --check

.PHONY: test
ntest: mypy ruff-verify


# Build

.PHONY: build
build:
	uv run trivianki build TriviAnki.apkg
