# Mandatory requirements
# install, run, debug, clean, lint, lint-strict

PYTHON	:= uv run python
SRC_DIR	:= src
RM		:= rm -rf

install: check-venv
	uv sync

run:
	$(PYTHON) -m $(SRC_DIR) 

debug:
	$(PYTHON) -m pdb -m $(SRC_DIR) 

clean:
	find . -name "*.pyc" -type f -delete -print
	find . -type d  -name "__pycache__" -delete -print
	$(RM) .mypy_cache
	$(RM) .pytest_cache
	$(RM) data/output/*

lint:
	uv run flake8 $(SRC_DIR)
	uv run mypy $(SRC_DIR) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 $(SRC_DIR)
	uv run mypy --strict $(SRC_DIR)

build: check-venv
	python3 -m build

.PHONY: install run debug clean lint lint-strict build
