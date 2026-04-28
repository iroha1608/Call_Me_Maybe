import sys
# import os
import json
from pathlib import Path
from typing import Any

from src.cli_arg import parse_arguments, CLIConfig
from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints import ConstraintFilter
from src.engine import GenerationEngine
from src.models import FunctionCallResult, FunctionDefinition


def _load_json_file(file_path: str) -> Any:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("Required file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {path}: {e}") from e


def _build_prompt(
    prompt_text: str, function_schema: list[FunctionDefinition]
) -> str:
    schema_str = json.dumps(function_schema, indent=2)
    return (
        "<|im_start|>system\n"
        "You are a strict JSON API. "
        "Convert the user's natural language request "
        "into a Json function call.\n"
        "Do NOT output function definitions, schemas, or types."
        "You MUST outputsctual values (numbers, strings) "
        "based on the user's request.\n"
        "Available functions:\n"
        f"{schema_str}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "What is the sum of 12 and 8?\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        '{\n'
        '    "name": "fn_add_numbers",\n'
        '    "parameters": {\n'
        '        "a": 12.0,\n'
        '        "b": 8.0\n'
        '        }\n'
        '    }\n'
        '}'
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _save_json_file(file_path: str, results: Any) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except IOError as e:
        raise IOError(f"Failed to write output to {path}: {e}") from e


def main() -> None:
    try:
        # CLIから指定された(またはデフォルトの)各データの読み込み->config
        config: CLIConfig = parse_arguments()

        # プロンプトの読み込み->prompts_data
        prompts_data: list[dict[str, str]] = _load_json_file(config.input)
        if not isinstance(prompts_data, list):
            raise ValueError(
                "Input prompts file must contain a JSON array"
            )

        # 関数データの読み込み->functions_data
        functions_data: list[FunctionDefinition] = (
            _load_json_file(config.function_definition)
        )
        if not isinstance(functions_data, list):
            raise ValueError(
                "Function definitons file must contain a JSON array"
            )

        print("1. Initializing models and components...", file=sys.stderr)

        # 依存オブジェクトの構築
        llm_client = LLMClient()
        tokenizer = Tokenizer(llm_client)
        constraint_filter = ConstraintFilter(
            tokenizer=tokenizer,
            available_functions=functions_data
        )
        engine = GenerationEngine(
            llm_client=llm_client,
            tokenizer=tokenizer,
            constraint_filter=constraint_filter
        )

        print(f"2. Starting processing of {len(prompts_data)} prompts...",
              file=sys.stderr)

        results: list[dict[str, Any]] = []
        # prompts_dataからpromptを一つづつ処理
        for index, prompt in enumerate(prompts_data, start=1):
            try:
                prompt_text = prompt["prompt"]
            except KeyError:
                continue
            print(f"3. DEBUG: prompt='{prompt_text}'")
            # 取り出したpromptにkey="prompt"がなければ飛ばす
            if not prompt_text:
                print(
                    f"Waring: skipping prompt {index} "
                    f"due to missing 'prompt' key",
                    file=sys.stderr
                )
                continue

            # コンテキストを含めたプロンプトの構築(アリ?)
            primed_prompt = _build_prompt(prompt_text, functions_data)

            try:
                # 推論エンジンの実行
                output_data = engine.generate(primed_prompt)
                # output_data = engine.generate(prompt_text)

                # 結果の記録(要件に準拠したformat)
                result_model = FunctionCallResult(
                    prompt=prompt_text,
                    name=output_data.get("name", "unknown"),
                    parameters=output_data.get("parameters", {})
                )
                # model.dump() -> dict形式での保存
                results.append(result_model.model_dump())
            except Exception as e:
                print(f"Error generating response for prompt"
                      f"'{prompt_text}': {e}", file=sys.stderr)
                results.append({
                    "prompt": prompt_text,
                    "error": str(e)
                })

        # 結果の保存
        _save_json_file(config.output, results)
        print(f"Success: Processed {len(results)} items. "
              f"Results saved to {config.output}", file=sys.stderr)
    except Exception as e:
        print(f"MainError: Pipeline execution failed."
              f"{e}", file=sys.stderr)
        print(e.__doc__)
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt as e:
        print(f"KeyboardINterruptError: {e}", file=sys.stderr)
        sys.exit(1)
