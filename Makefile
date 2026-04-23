UV		:= uv
PYTHON	:= python -m
SRC_DIR	:= src
RM		:= rm -rf

# Mandatory requirements
# install, run, debug, clean, lint, lint-strict

setup:
	$(UV) venv

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) $(SRC_DIR) 

debug:
	$(UV) run $(PYTHON) pdb -m $(SRC_DIR) 

clean:
	find . -name "*.pyc" -type f -delete -print
	find . -type d -name "__pycache__" -delete -print
	$(RM) .mypy_cache
	$(RM) .pytest_cache
	$(RM) data/output/*

lint:
	$(UV) run flake8 $(SRC_DIR)
	$(UV) run mypy $(SRC_DIR) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
	$(UV) run ruff $(SRC_DIR)

lint-strict:
	$(UV) run flake8 $(SRC_DIR)
	$(UV) run mypy --strict $(SRC_DIR)

test:
	$(UV) run pytest tests

build: check-venv
	$(PYTHON) build

.PHONY: install run debug clean lint lint-strict build
