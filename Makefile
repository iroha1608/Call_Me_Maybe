UV		:= uv
PYTHON	:= python3 -m
SRC_DIR	:= src
RM		:= rm -rf

# Mandatory requirements
# install, run, debug, clean, lint, lint-strict

install:
	$(UV) sync

# uvのインストール
uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

# 仮想環境のセットアップ
setup:
	$(UV) venv

# 課題に必要なファイルのインストール
llm:
	$(UV) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(UV) pip install transformers huggingface_hub accelerate

run:
	$(UV) run $(PYTHON) $(SRC_DIR)

debug:
	$(UV) run $(PYTHON) pdb -m $(SRC_DIR)

clean:
	find . -name "*.pyc" -type f -delete -print
	find . -type d -name "__pycache__" -delete -print
	$(RM) .mypy_cache
	$(RM) .pytest_cache
	$(RM) .ruff_cache
	$(RM) data/output/*

fclean: clean
	$(RM) .venv

lint:
	- $(UV) run flake8 $(SRC_DIR)
	- $(UV) run mypy $(SRC_DIR) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	- $(UV) run ruff check $(SRC_DIR)
	- $(UV) run ty check $(SRC_DIR)

lint-strict:
	$(UV) run flake8 $(SRC_DIR)
	$(UV) run mypy --strict $(SRC_DIR)

test:
	$(UV) run pytest tests

build:
	$(UV) run $(PYTHON) build

.PHONY: install run debug clean lint lint-strict build uv setup llm fclean
