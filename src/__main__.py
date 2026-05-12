import sys
# import os
import json
import time
import re
from pathlib import Path
from typing import Any
from pydantic import ValidationError

from src.cli_arg import parse_arguments, CLIConfig
from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints.filter import ConstraintFilter
from src.engine import GenerationEngine
from src.models import PromptInput, FunctionDefinition, FunctionCallResult


def _load_json_file(file_path: str) -> Any:
    """JSONファイルの読み込み"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError as e:
        raise ValueError(f"Required file not found: {path}") from e

    except PermissionError as e:
        raise ValueError(f"Permission denied: {path}") from e

    except IsADirectoryError as e:
        raise ValueError(f"Path is a derectory, not a file: {path}") from e

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {path}: {e}") from e


def _calculate_jaccard_similarity(
    user_prompt: str, fn_def: FunctionDefinition
) -> float:
    """プロンプトと関数定義間のJaccard係数を計算する。"""
    # 正規表現で英数字のみ抽出
    prompt_words = set(re.findall(r'[a-zA-Z0-9_]+', user_prompt.lower()))
    name = fn_def.name
    desc = fn_def.description

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
    """LLMに渡すメインプロンプトの生成。"""
    scored_functions = [
        (fn, _calculate_jaccard_similarity(user_prompt, fn))
        for fn in function_schema
    ]
    scored_functions.sort(key=lambda x: x[1], reverse=True)
    top_functions = [fn for fn, score in scored_functions[:2]]

    # Json -> mdへ変換
    markdown_schema = ""
    for fn in top_functions:
        markdown_schema += f"- Function Name: {fn.name}\n"
        markdown_schema += f"  Purpose: {fn.description}\n"
        if fn.parameters:
            markdown_schema += "  Arguments:\n"
            for prop_name, prop_details in fn.parameters.items():
                markdown_schema += f"    * {prop_name} ({prop_details.type})\n"
        markdown_schema += "\n"

    # Main Prompt
    prompt = (
        "<|im_start|>system\n"
        "You are a strict Data Extraction Engine.\n"
        "Your ONLY role is to act as a copy-and-paste tool between "
        "the user's text and the JSON parameters.\n\n"
        "[Execution Steps]\n"
        "Step 1 (Read): Read the available functions and "
        "the user's input text.\n"
        "Step 2 (Select): Select the correct function and read its required "
        "parameters and types.\n"
        "Step 3 (Extract): Find the exact information in the user's text "
        "that matches each parameter.\n"
        "Step 4 (Copy): Copy the extracted information directly "
        "into the JSON format.\n\n"
        "[Extraction Rules]\n"
        "- Numbers: Extract exact numbers from the text. "
        "Never invent, calculate, or output default numbers like 0 or -1.\n"
        "Even if the parameter type is 'number', YOU must determine whether "
        "it should be an integer or a float based on the context. "
        "Do not add unnecessary decimals.\n"
        "- Words/Sentences: Copy the target text EXACTLY. "
        "Do not summarize or truncate.\n"
        "- Patterns/Symbols: If asked to replace with a symbol "
        "(e.g., 'asterisks'), output the actual symbol (e.g, '\*'.) "
        "If a regex pattern is needed, "
        "output the standard regex (e.g., \\d+).\n\n"
        "Available functions:\n"
        f"{markdown_schema}"
        "<|im_end|>\n"
    )
    prompt += f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n{\n"

    return prompt


def _save_json_file(file_path: str, results: Any) -> None:
    """出力結果をJSONに保存する関数"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except IOError as e:
        raise IOError(f"Failed to write output to {path}: {e}") from e


def main() -> None:
    program_start_time = time.time()
    print("\x1b[2J\x1b[H\x1b[s", end="")
    print("1. ...")
    try:
        # -------------------- 外部データの読み込み --------------------
        # CLIから指定された(またはデフォルトの)各データの読み込み->config
        config: CLIConfig = parse_arguments()

        # -------------------- プロンプトのバリデーション --------------------
        # プロンプトの読み込み->raw_prompts_data
        raw_prompts_data: list[dict[str, Any]] = _load_json_file(config.input)
        if not isinstance(raw_prompts_data, list):
            raise ValueError(
                "Input prompts file must contain a JSON array"
            )

        valid_prompts_data: list[PromptInput] = []
        for index, p_data in enumerate(raw_prompts_data, start=1):
            try:
                valid_p = PromptInput.model_validate(p_data)
                valid_prompts_data.append(valid_p)
            except ValidationError as e:
                print(
                    f"[WARNING] Skipping prompt {index} "
                    f"due to validation error:\n{e}", file=sys.stderr)
                continue

        if not valid_prompts_data:
            raise ValueError(
                "The required keys were missing from all functions."
            )

        print(f"[\33[32mINFO\33[0m] {len(valid_prompts_data)} "
              "prompts data has been loaded!")
        time.sleep(0.3)

        # -------------------- 関数定義のバリデーション --------------------
        # 関数データの読み込み->raw_functions_data
        raw_functions_data = _load_json_file(config.function_definition)
        if not isinstance(raw_functions_data, list):
            raise ValueError(
                "Function definitons file must contain a JSON array."
            )

        valid_functions_data: list[FunctionDefinition] = []
        for index, f_data in enumerate(raw_functions_data, start=1):
            try:
                valid_fn = FunctionDefinition.model_validate(f_data)
                valid_functions_data.append(valid_fn)
            except ValidationError as e:
                print(
                    f"[\33[33mWARNING\33[0m] Skipping function {index} "
                    f"due to validation error:\n{e}", file=sys.stderr)
                continue

        if not valid_functions_data:
            raise ValueError(
                "The required keys were missing from all functions."
            )

        print(f"[\33[32mINFO\33[0m] {len(valid_functions_data)} "
              "functions definition has been loaded!")
        time.sleep(0.3)

        # -------------------- 依存オブジェクトの構築 --------------------
        print("2. \33[1mInitializing models and components...\33[0m")

        llm_client = LLMClient()
        print("[\33[32mINFO\33[0m] LLMClient has been loaded!")
        time.sleep(0.3)

        tokenizer = Tokenizer(llm_client)
        print("[\33[32mINFO\33[0m] Tokenizer has been loaded!")
        time.sleep(0.3)

        constraint_filter = ConstraintFilter(
            tokenizer=tokenizer,
            available_functions=valid_functions_data
        )
        print("[\33[32mINFO\33[0m] ConstraintFilter has been loaded!")
        time.sleep(0.3)

        engine = GenerationEngine(
            llm_client=llm_client,
            tokenizer=tokenizer,
            constraint_filter=constraint_filter
        )
        print("[\33[32mINFO\33[0m] Engine has been loaded!")
        time.sleep(0.3)

        # -------------------- プロンプトの読み込み --------------------
        print("3. Starting processing of "
              f"{len(valid_prompts_data)} prompts...")

        results: list[dict[str, Any]] = []

        # prompts_dataからpromptを一つづつ処理
        time.sleep(2)
        for index, prompt in enumerate(valid_prompts_data, start=1):

            raw_prompt: str = prompt.prompt
            # プロンプト中の'"'->'\"'に変換
            user_prompt = raw_prompt
            if '"' in raw_prompt:
                user_prompt = raw_prompt.replace('"', '\\"')

            print(f"Prompt{index}. '{raw_prompt}'")
            print("3", end="")
            time.sleep(1.2)
            print("2", end="")
            time.sleep(1.2)
            print("1...", end="")
            time.sleep(1.2)

            # 推論の直前でプロンプト固有の情報をFSMにセット
            # 内部情報をリセットする
            constraint_filter.set_user_prompt(raw_prompt)

            # コンテキストを含めたプロンプトの構築
            primed_prompt = _build_prompt(user_prompt, valid_functions_data)

            try:
                # 推論エンジンの実行(+時間計測)
                prompt_start_time = time.time()
                output_data = engine.generate(primed_prompt)
                prompt_end_time = time.time()

                # 結果の記録(要件に準拠したフォーマットか判定)
                result_model = FunctionCallResult(
                    prompt=raw_prompt,
                    name=output_data.get("name", "unknown"),
                    parameters=output_data.get("parameters", {})
                )

                # model.dump() -> dict形式での保存、resultsに追加していく
                results.append(result_model.model_dump())
                p_time = prompt_end_time - prompt_start_time
                print(f"Prompt{index}. {p_time:.4f} seconds")
                print(f"Prompt{index}. '{user_prompt}'")
                time.sleep(3)

            except Exception as e:
                print(f"Error generating response for prompt"
                      f"'{user_prompt}': {e}", file=sys.stderr)
                results.append({
                    "prompt": raw_prompt,
                    "error": str(e)
                })

        # ------------------------- 結果の保存 -------------------------
        _save_json_file(config.output, results)
        print(f"4. Success: Processed {len(results)} items. "
              f"Results saved to {config.output}")
        program_end_time = time.time()
        t_time = program_end_time - program_start_time
        print(f"[\33[32mINFO\33[0m] Total. {t_time:.4f} seconds")

    except Exception as e:
        print(f"[\33[31mMainError\33[0m]: Pipeline execution failed. {e}",
              file=sys.stderr)
        print(e.__doc__)
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[\33[31mError\33[0m]: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("KeyboardInterruptError: "
              "Ctrl + c has been detected.", file=sys.stderr)
        sys.exit(1)
