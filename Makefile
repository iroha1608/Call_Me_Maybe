# Mandatory requirements
# install, run, debug, clean, lint, lint-strict

check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "Error: Please run this within a virtual environment."; \
		exit 1; \
	fi

install: check-venv
	pip install -r requirements.txt

run:
	python3 a_maze_ing.py config.txt

debug:
	python3 -m pdb a_maze_ing.py config.txt

clean:
	find . -name "*.pyc" -type f -delete -print
	find . -type d  -name "__pycache__" -delete -print
	rm -rf .mypy_cache
	rm -rf dist/
	rm -rf mazegen.egg-info/

lint:
	-flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	-flake8 .
	mypy . --strict

build: check-venv
	python3 -m build

.PHONY: check-venv install run debug clean lint lint-strict build
