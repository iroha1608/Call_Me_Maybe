import sys
import os
import json
from typing import Any

from src.cli import parse_arguments
from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints import ConstraintFilter
from src.engine import GenerationEngine
from src.models import FunctionCallResult


def main() -> None:
    try:
        # configの読み込み
        config = parse_arguments()
        # プロンプトの読み込み
        with open(config.input, "r", encoding="utf-8") as f_in:
            prompts_data: list[dict[str, str]] = json.load(f_in)
            if not isinstance(prompts_data, list):
                raise ValueError(
                    "Input prompts file must contain a JSON array"
                )
        # 関数データの読み取り
        with open(config.function_definiton, "r", encoding="utf-8") as f_func:
            functions_data = json.load(f_func)
            if not isinstance(functions_data, list):
                raise ValueError(
                    "Function definitons file must contain a JSON array"
                )

        # 依存オブジェクトの構築
        llm_client = LLMClient()
        tokenizer = Tokenizer()
        constraint_filter = ConstraintFilter(
            tokenizer=tokenizer,
            available_functions=functions_data
        )
        engine = GenerationEngine(
            llm_client=llm_client,
            tokenizer=tokenizer,
            constraint_filter=constraint_filter
        )

        results: list[dict[str, Any]] = []
        for item in prompts_data:
            prompt_text = item.get("prompt")
            if not prompt_text:
                continue

            output_data = engine.generate(prompt_text)

            result_model = FunctionCallResult(
                prompt=prompt_text,
                name=output_data.get("name", "unknown"),
                parameters=output_data.get("parameters", {})
            )
            results.append(result_model.model_demp())

        output_dir = os.path.dirname(config.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(config.output, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"MainError: Pipeline execution failed."
              f"{e}", file=sys.stderr)
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
