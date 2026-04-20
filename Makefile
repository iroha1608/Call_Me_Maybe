UV		:= uv run
PYTHON	:= python -m
SRC_DIR	:= src
RM		:= rm -rf

# Mandatory requirements
# install, run, debug, clean, lint, lint-strict

install: check-venv
	uv sync

run:
	$(UV) $(PYTHON) $(SRC_DIR) 

debug:
	$(UV) $(PYTHON) pdb -m $(SRC_DIR) 

clean:
	find . -name "*.pyc" -type f -delete -print
	find . -type d -name "__pycache__" -delete -print
	$(RM) .mypy_cache
	$(RM) .pytest_cache
	$(RM) data/output/*

lint:
	$(UV) flake8 $(SRC_DIR)
	$(UV) mypy $(SRC_DIR) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(UV) flake8 $(SRC_DIR)
	$(UV) mypy --strict $(SRC_DIR)

build: check-venv
	$(PYTHON) build

.PHONY: install run debug clean lint lint-strict build
