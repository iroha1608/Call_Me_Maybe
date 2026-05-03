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
    except FileNotFoundError as e:
        raise ValueError(f"Required file not found: {path}") from e

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

    name_words = set(
        re.findall(r'[a-zA-Z0-9]+', name.lower().replace("_", " "))
    )
    desc_words = set(re.findall(r'[a-zA-Z0-9_]+', desc.lower()))
    target_words = desc_words | name_words

    intersection = prompt_words & target_words
    union = prompt_words | target_words
    return len(intersection) / len(union) if union else 0.0


def _build_prompt(
    user_prompt: str, function_schema: list[FunctionDefinition]
) -> str:
    scored_functions = [
        (fn, _calculate_jaccard_similarity(user_prompt, fn))
        for fn in function_schema
    ]
    scored_functions.sort(key=lambda x: x[1], reverse=True)
    top_functions = [fn for fn, score in scored_functions[:2]]

    # 動的Few-shot, 丸暗記ではないモデルの学習
    # 正規表現、引数のマッピング規則の学習
    examples: dict[str, list[Any]] = {
        'fn_add_numbers': [(
                'What is the sum of 10 and 5?',
                '{"name": "fn_add_numbers", '
                '"parameters": {"a": 10, "b": 5}}'),
            ('Add 42 and 8',
                '{"name": "fn_add_numbers", '
                '"parameters": {"a": 42, "b": 8}}')
        ],
        'fn_greet': [(
            'Greet emily',
            '{"name": "fn_greet", '
            '"parameters": {"name": "emily"}}'),
            ('Say hello to captain',
                '{"name": "fn_greet", '
                '"parameters": {"name": "captain"}}')
        ],
        'fn_reverse_string': [(
            'Reverse the string \'apple\'',
            '{"name": "fn_reverse_string", '
            '"parameters": {"s": "apple"}}'),
            ('Reverse "XYZ_123"',
                '{"name": "fn_reverse_string", '
                '"parameters": {"s": "XYZ_123"}}')
        ],
        'fn_get_square_root': [(
            'What is the square root of 9?',
            '{"name": "fn_get_square_root", '
            '"parameters": {"a": "9"}}'),
            ('Calculate the square root of 144.0',
                '{"name": "fn_get_square_root", '
                '"parameters": {"a": "144.0"}}')
        ],
        'fn_substitute_string_with_regex': [(
            'Replace all numbers in "I have 2 apples" with NUMBERS',
            '{"name": "fn_substitute_string_with_regex", '
            '"parameters": {'
            '"source_string": "I have 2 apples", '
            '"regex": "\\\\d+", '
            '"replacement": "NUMBERS"}}'),
            ('Replace all vowels in "Testing 123 is important" with asterisks',
                '{"name": "fn_substitute_string_with_regex", '
                '"parameters": {'
                '"source_string": "Testing 123 is important", '
                '"regex": "[aeiouAEIOU]", '
                '"replacement": "*"}}'),
            ('Substitute "apple" with "orange" in '
             '"I ate an apple and another apple"',
                '{"name": "fn_substitute_string_with_regex", '
                '"parameters": {'
                '"source_string": "I ate an apple and another apple", '
                '"regex": "apple", '
                '"replacement": "orange"}}'),
            ('Change the word "sad" to "happy" in '
             '"The boy was sad because his toy broke"',
                '{"name": "fn_substitute_string_with_regex", '
                '"parameters": {'
                '"source_string": "The boy was sad because his toy broke", '
                '"regex": "sad", '
                '"replacement": "happy"}}'),
            ('Replace the letter "s" with "z" in '
             '"This is a very long sentence that should not be cut short."',
                '{"name": "fn_substitute_string_with_regex", '
                '"parameters": {'
                '"source_string": "This is a very long sentence '
                'that should not be cut short.", '
                '"regex": "s", '
                '"replacement": "z"}}')
        ]
    }

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

    # Main Prompt
    prompt = (
        "<|im_start|>system\n"
        "You are a strict AI assistant designed for function calling.\n"
        "Your task is to analyze the user's prompt "
        "and generate j JSON object to call the function.\n\n"
        "CRITICAL RULES:\n"
        "1. Extract values exactly based on the function's arguments.\n"
        "2. When an argument requires a full sentence or target text, "
        "DO NOT truncate it. Copy the entire string exactly.\n"
        "3. When an argument requires a pattern, rule, or specific target, "
        "output the exact word or generate standard formats "
        "(e.g., standard Regex like \\d+).\n"
        "Available functions:\n"
        f"{markdown_schema}"
        "<|im_end|>\n"
    )
    # 最も関連性の高い関数の具体例を抽出
    top_fn_name = _get_attr(top_functions[0], "name", "")
    no_data_text = [
        ("Extract data", '{"name": "unknown", "parameters": {}}')
    ]
    fn_examples = examples.get(top_fn_name, no_data_text)

    # さらに関連度の高い1例を取得
    if len(fn_examples) > 1:
        prompt_words = set(re.findall(r'[a-zA-Z0-9]+', user_prompt.lower()))
        scored_examples = []
        for ex_u, ex_a in fn_examples:
            ex_words = set(re.findall(r'[a-zA-Z0-9]+', ex_u.lower()))
            intersection = prompt_words & ex_words
            union = prompt_words | ex_words
            score = len(intersection) / len(union) if union else 0.0
            scored_examples.append((score, (ex_u, ex_a)))
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        fn_examples = [ex for score, ex in scored_examples[:2]]

    # 関数ごとの具体例を注入
    # for ex_u, ex_a in fn_examples:
        # prompt += f"<|im_start|>user\n{ex_u}\n<|im_end|>\n"
        # prompt += f"<|im_start|>assistant\n{ex_a}\n<|im_end|>\n"
        # print(f"DEBUG: user='{ex_u}'\nassist='{ex_a}'")
    prompt += f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n{\n"

    return prompt


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
