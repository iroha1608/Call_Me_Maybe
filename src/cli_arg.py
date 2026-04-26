import sys
import argparse
from pydantic import BaseModel, Field


class CLIConfig(BaseModel):
    function_definition: str = Field(
        default="data/input/functions_definition.json",
        description="Path to the function definitions JSON file."
    )
    input: str = Field(
        default="data/input/function_calling_tests.json",
        description="Path to the input prompts JSON file"
    )
    output: str = Field(
        default="data/output/function_calling_results.json",
        description="Path to the output results JSON file"
    )


def parse_arguments() -> CLIConfig:
    parser = argparse.ArgumentParser(
        description="Introduction to function calling in LLMs"
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output results JSON file"
    )

    try:
        # 解析
        args = parser.parse_args()
        # 型検証、安全なデータモデルの生成
        return CLIConfig(
            function_definition=args.functions_definition,
            input=args.input,
            output=args.output
        )

    except Exception as e:
        print(f"CLI Error: Argument parsing failed."
              f"{e}", file=sys.stderr)
        exit(1)
