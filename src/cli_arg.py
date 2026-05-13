import sys
import argparse
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError


class CLIConfig(BaseModel):
    """
        Pathを使用しWindows、Mac/Linuxのパス形式を補完。
    """
    function_definition: Path = Field(
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
        if v.suffix.lower() != '.json':
            raise ValueError(f"File must have a '.json' extension: {v}")
        return v


def parse_arguments() -> CLIConfig:
    """
        aragparseは自動でSystemExit、
        その他のパースはpydanticでバリデーションチェック
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

        # argparseでNoneになったものを除外して辞書化
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
