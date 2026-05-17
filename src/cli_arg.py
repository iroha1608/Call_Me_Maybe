"""
    Command-line interface argument parsing
    and validation for the function calling application.
"""
import sys
import argparse
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError


class CLIConfig(BaseModel):
    """
        Configuration model for CLI arguments.
    """
    functions_definition: Path = Field(
        default=Path("data/input/functions_definition.json"),
        description="Path to the function definitions JSON file."
    )
    input: Path = Field(
        default=Path("data/input/function_calling_tests.json"),
        description="Path to the input prompts JSON file"
    )
    output: Path = Field(
        default=Path("data/output/function_calling_results.json"),
        description="Path to the output results JSON file"
    )

    # ユーザーがどのフラグにどんな文字列を入れても必ず.jsonで終わるかチェック
    @field_validator("*")
    def check_json_extension(cls, v: Path) -> Path:
        """
            Validate that the provided path has a .json extension.
            Args:
                v (Path): The path to validate.
            Returns:
                Path: The validated path.
            Raises:
                ValueError: If the file does not have a .json extension.
        """
        if v.suffix.lower() != '.json':
            raise ValueError(f"File must have a '.json' extension: {v}")
        return v


def parse_arguments() -> CLIConfig:
    """
        Parse command-line arguments and return a validated CLIConfig instance.
        Returns:
            CLIConfig: The validated configuration object
                            containing the parsed arguments.
        Raises:
            ValueError:
                If any of the provided paths do not have a .json extension.
            ValidationError: If the provided arguments
                            do not conform to the CLIConfig model.
        """
    parser = argparse.ArgumentParser(
        description="Introduction to function calling in LLMs"
    )

    parser.add_argument(
        "-f", "--functions_definition",
        type=str,
        help="Path to the function definitions JSON file."
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to the input prompts JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to the output results JSON file"
    )

    try:
        # 解析、不正な引数は自動でSystemExitが呼ばれhelpが出る
        args = parser.parse_args()

        kwargs = {k: v for k, v in vars(args).items() if v is not None}

        # 型検証、安全なデータモデルの生成
        return CLIConfig(**kwargs)

    except ValidationError as e:
        print("[\33[31mCLI Error\33[0m]: Argument validation failed.\n"
              f"{e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"[\33[31mCLI Error\33[0m]: Unexpected error during parsing: \n"
              f"{e}", file=sys.stderr)
        sys.exit(1)
