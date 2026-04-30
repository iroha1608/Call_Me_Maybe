import sys
# import os
import json
import time
import re
from pathlib import Path
from typing import Any

from src.cli_arg import parse_arguments, CLIConfig
from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints import ConstraintFilter
from src.engine import GenerationEngine
from src.models import FunctionCallResult, FunctionDefinition


def _get_attr(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _load_json_file(file_path: str) -> Any:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("Required file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {path}: {e}") from e


def _calculate_jaccard_similarity(
    user_prompt: str, fn_def: FunctionDefinition
) -> float:
    """プロンプトと関数定義間のJaccard係数を計算する。"""
    # 正規表現で英数字のみ抽出
    prompt_words = set(re.findall(r'[a-zA-Z0-9_]+', user_prompt.lower()))
    name = _get_attr(fn_def, "name", "")
    desc = _get_attr(fn_def, "description", "")

    name_words = set(name.lower().replace("_", " ").split())
    desc_words = set(re.findall(r'[a-zA-Z0-9_]+', desc.lower()))
    target_words = desc_words | name_words

    intersection = prompt_words & target_words
    union = prompt_words | target_words
    return len(intersection) / len(union) if union else 0.0


def _build_prompt(
    user_prompt: str, function_schema: list[FunctionDefinition]
) -> str:
    # 関連性の高い上位2つの関数に絞り込む
    scored_functions = [
        (fn, _calculate_jaccard_similarity(user_prompt, fn))
        for fn in function_schema
    ]
    scored_functions.sort(key=lambda x: x[1], reverse=True)
    top_functions = [fn for fn, score in scored_functions[:2]]

    # 動的Few-shotの定義
    # 正規表現、引数のマッピング規則の学習
    examples = {
            'fn_add_numbers': 'User: What is the sum of 10 and 5?\n'
            'Assistant:\n'
            '{"name": "fn_add_numbers", '
            '"parameter": {"a": 10.0, "b": 5.0}}',
            'fn_greet': 'User: Greet Alice\n'
            'Assistant:\n'
            '{"name": "fn_greet", '
            '"parameter": {"name": "Alice"}}',
            'fn_reverse_string': 'User: Reverse the string \'world\'\n'
            'Assistant:\n'
            '{"name": "fn_reverse_string", '
            '"parameter": {"s": "world"}}',
            'fn_get_square_root': 'User: What is the square root of 9?\n'
            'Assistant:\n'
            '{"name": "fn_get_square_root", '
            '"parameter": {"a": "9.0"}}',
            'fn_substitute_string_with_regix':
            'User: Replace all numbers in "I have 2 apples" with X\n'
            'Assistant:\n'
            '{"name": "fn_substitute_string_with_regix", '
            '"parameter": {'
            '"source_string": "I have 2 apples", '
            '"regix": "\\\\d+", '
            '"replacement": "X"}}'
    }

    # 最も関連性の高い関数の具体例を抽出
    top_fn_name = _get_attr(top_functions[0], "name", "")
    no_data_text: str = (
        'User: Extract data\nAssistant:\n'
        '{"name": "unknown", "parameters": {}}'
    )
    few_shot_example = examples.get(top_fn_name, no_data_text)

    # Json -> mdへ変換
    markdown_schema = ""
    for fn in top_functions:
        name = _get_attr(fn, "name", "")
        desc = _get_attr(fn, "description", "")
        markdown_schema += f"- Function Name: {name}\n"
        markdown_schema += f"  Purpose: {desc}\n"
        params = _get_attr(fn, "parameters", {})
        if params and isinstance(params, dict):
            markdown_schema += "  Arguments:\n"
            for prop_name, prop_details in params.items():
                if isinstance(prop_details, dict):
                    prop_type = _get_attr(prop_details, "type", "any")
                else:
                    prop_type = "any"
                markdown_schema += f"    * {prop_name} ({prop_type})\n"
        markdown_schema += "\n"

    return (
        "<|im_start|>system\n"
        "You are a strict data extraction engine. "
        "Extract exact values from the user's input "
        "based on the provided functions.\n"
        "Output ONLY a valid JSON object. "
        "Do not output anything else.\n\n"
        "Example:\n"
        f"{few_shot_example}"
        "Available functions:\n"
        f"{markdown_schema}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "{\n"
    )
    # schema_str = json.dumps(function_schema, indent=2)


def _save_json_file(file_path: str, results: Any) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except IOError as e:
        raise IOError(f"Failed to write output to {path}: {e}") from e


def main() -> None:
    program_start_time = time.time()
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
        print("DEBUG: LLMClient define")
        tokenizer = Tokenizer(llm_client)
        print("DEBUG: Tokenizer define")
        constraint_filter = ConstraintFilter(
            tokenizer=tokenizer,
            available_functions=functions_data
        )
        print("DEBUG: ConstraintFilter define")
        engine = GenerationEngine(
            llm_client=llm_client,
            tokenizer=tokenizer,
            constraint_filter=constraint_filter
        )
        print("DEBUG: Engine define")

        print(f"2. Starting processing of {len(prompts_data)} prompts...",
              file=sys.stderr)

        results: list[dict[str, Any]] = []
        # prompts_dataからpromptを一つづつ処理
        for index, prompt in enumerate(prompts_data, start=1):
            if not isinstance(prompt, dict):
                continue
            try:
                user_prompt = prompt["prompt"]
            except KeyError:
                continue
            print(f"DEBUG: Prompt{index}. '{user_prompt}'", file=sys.stderr)
            # 取り出したpromptにkey="prompt"がなければ飛ばす
            if not user_prompt:
                print(
                    f"Waring: skipping prompt {index} "
                    f"due to missing 'prompt' key",
                    file=sys.stderr
                )
                continue
            # 推論の直前でプロンプト固有の情報をFSMにセット
            # 内部情報をクリアにする
            constraint_filter.set_user_prompt(user_prompt)

            # コンテキストを含めたプロンプトの構築
            primed_prompt = _build_prompt(user_prompt, functions_data)

            try:
                # 推論エンジンの実行
                prompt_start_time = time.time()
                output_data = engine.generate(primed_prompt)
                prompt_end_time = time.time()

                # 結果の記録(要件に準拠したformat)
                result_model = FunctionCallResult(
                    prompt=user_prompt,
                    name=output_data.get("name", "unknown"),
                    parameters=output_data.get("parameters", {})
                )

                # model.dump() -> dict形式での保存
                results.append(result_model.model_dump())
                p_time = prompt_end_time - prompt_start_time
                print(f"DEBUG: Prompt{index}. {p_time:.4f} seconds")
                print(f"DEBUG: Prompt{index}. '{user_prompt}'",
                      file=sys.stderr)
            except Exception as e:
                print(f"Error generating response for prompt"
                      f"'{user_prompt}': {e}", file=sys.stderr)
                results.append({
                    "prompt": user_prompt,
                    "error": str(e)
                })

        # 結果の保存
        _save_json_file(config.output, results)
        print(f"Success: Processed {len(results)} items. "
              f"Results saved to {config.output}", file=sys.stderr)
        program_end_time = time.time()
        t_time = program_end_time - program_start_time
        print(f"DEBUG: Total. {t_time:.4f} seconds")
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
