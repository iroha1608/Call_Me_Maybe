"""
    Main module for the function calling pipeline.
    This module orchestrates the entire process of reading input prompts,
    validating them, loading function definitions, initializing the necessary
    components (LLM client, tokenizer, constraint filter,
                                                    and generation engine),
    and executing the generation process for each prompt.
    It also handles the saving of results
    and error handling throughout the pipeline.
"""
import sys
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


def _load_json_file(file_path: Path) -> Any:
    """
        Load a JSON file and return its contents.
        Args:
            file_path (Path): The path to the JSON file.
        Returns:
            Any: The contents of the JSON file.
        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If the file cannot be decoded as UTF-8.
            PermissionError: If the file cannot be accessed due to permissions.
            IsADirectoryError: If the path is a directory, not a file.
            json.JSONDecodeError: If the file is not valid JSON.
            OSError: For other OS-related errors.
    """
    path = file_path

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError as e:
        raise ValueError(f"Required file not found: {path}: {e}") from e

    except UnicodeDecodeError as e:
        raise ValueError(
            f"File encoding error (MUST be UTF-8): {path}: {e}") from e

    except PermissionError as e:
        raise ValueError(f"Permission denied: {path}") from e

    except IsADirectoryError as e:
        raise ValueError(f"Path is a directory, not a file: {path}") from e

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {path}: {e}") from e

    except OSError as e:
        raise ValueError(f"OS error occurred while reading {path}: {e}") from e


def _calculate_jaccard_similarity(
    user_prompt: str, fn_def: FunctionDefinition
) -> float:
    """
        Calculate the Jaccard similarity
        between the user prompt and a function definition.
        Args:
            user_prompt (str): The user's input prompt.
            fn_def (FunctionDefinition):
                The function definition to compare against.
        Returns:
            float: The Jaccard similarity score (0.0 to 1.0).
    """
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
    """
        Build the prompt for the LLM
        based on the user input and available function definitions.
        Args:
            user_prompt (str): The user's input prompt.
            function_schema (list[FunctionDefinition]):
                A list of available function definitions.
        Returns:
            str: The constructed prompt to be sent to the LLM.
    """
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
        "You are a strict data extraction engine. "
        "Extract exact values from the user's input "
        "based on the provided functions.\n"
        "Available functions:\n"
        f"{markdown_schema}"
        "<|im_end|>\n"
    )
    prompt += f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n{\n"

    return prompt


def _save_json_file(file_path: Path, results: Any) -> None:
    """
        Save results to a JSON file.
        Args:
            file_path (Path): The path to the output JSON file.
            results (Any): The data to be saved in JSON format.
        Raises:
            PermissionError:
                If there is a permission issue when writing to the file.
            IOError: If there is an error writing to the file.
    """
    path = file_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    except PermissionError as e:
        raise IOError(f"Permission denied when writing to {path}: {e}") from e

    except IOError as e:
        raise IOError(f"Failed to write output to {path}: {e}") from e


def main() -> None:
    """Main function to execute the function calling pipeline."""
    program_start_time = time.time()
    print("\x1b[2J\x1b[H\x1b[s", end="")
    print("1. \33[1mLet's get the “Call Me Maybe” program started\33[0m")
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
        raw_functions_data: list[dict[str, Any]] = (
            _load_json_file(config.functions_definition))
        if not isinstance(raw_functions_data, list):
            raise ValueError(
                "Function definitions file must contain a JSON array."
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
        print("3. \33[1mStarting processing of "
              f"{len(valid_prompts_data)} prompts...\33[0m")

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
            print("3")
            time.sleep(1.2)
            print("2")
            time.sleep(1.2)
            print("1...")
            time.sleep(1.2)

            # 推論の直前でプロンプト固有の情報をFSMにセット
            # 内部情報をリセットする
            constraint_filter.set_user_prompt(raw_prompt)

            # コンテキストを含めたプロンプトの構築
            primed_prompt = _build_prompt(user_prompt, valid_functions_data)

            # 推論中のエラーは途中で止めずに次のプロンプトに継続する
            try:
                # 推論エンジンの実行(+時間計測)
                prompt_start_time = time.time()
                output_data = engine.generate(primed_prompt)
                prompt_end_time = time.time()

                # 結果の記録(要件に準拠したフォーマットか判定)
                if not isinstance(output_data, dict):
                    raise ValueError(
                        f"Engine returned invalid type: {type(output_data)}")
                result_model = FunctionCallResult(
                    prompt=raw_prompt,
                    name=output_data["name"],
                    parameters=output_data["parameters"]
                )

                # model.dump() -> dict形式での保存、resultsに追加していく
                results.append(result_model.model_dump())
                p_time = prompt_end_time - prompt_start_time
                print(f"Prompt{index}. {p_time:.4f} seconds")
                print(f"Prompt{index}. '{user_prompt}'")
                time.sleep(3)

            # JSON必須キー不足、型違い
            except (KeyError, ValidationError) as e:
                print(f"[\33[31mError\33[0m] Invalid output format for prompt "
                      f"'{user_prompt}': {e}", file=sys.stderr)
                results.append({
                    "prompt": raw_prompt,
                    "error": f"Format error: {e}"
                })

            # ネットワーク、タイムアウト、エンジンのクラッシュ
            except Exception as e:
                print(f"[\33[31mError\33[0m] System failure during prompt "
                      f"'{user_prompt}': {e}", file=sys.stderr)
                results.append({
                    "prompt": raw_prompt,
                    "error": f"System error: {e}"
                })

        # ------------------------- 結果の保存 -------------------------
        _save_json_file(config.output, results)
        print(f"4. \33[1mSuccess: Processed {len(results)} items. "
              f"Results saved to {config.output}\33[0m")
        program_end_time = time.time()
        t_time = program_end_time - program_start_time
        print(f"[\33[32mINFO\33[0m] Total. {t_time:.4f} seconds")

    # Ctrl + Cでの中断もそこまでの結果を保存する
    except KeyboardInterrupt:
        print("[\33[31mWarning\33[0m]: Process interrunpted by user. "
              "Saving partial results...", file=sys.stderr)
        if "results" in locals() and results:
            _save_json_file(config.output, results)
            print(f"Partial results saved to {config.output}", file=sys.stderr)
        sys.exit(1)

    # 可能な限り保存
    except Exception as e:
        print(f"[\33[31mMainError\33[0m]: Pipeline execution failed. {e}",
              file=sys.stderr)
        print(e.__doc__)
        if "results" in locals() and results:
            _save_json_file(config.output, results)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[\33[31mError\33[0m]: {e}", file=sys.stderr)
        sys.exit(1)
