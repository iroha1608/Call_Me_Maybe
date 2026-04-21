import sys

from src.cli import parse_arguments
from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints import ConstrainFilter
from src.engine import GenerationEngine
from src.models import FunctionCallResult


def main() -> None:
    try:
        # configの読み込み->プロンプト、関数データの読み込み
        config = parse_arguments()

        with open(config.input, "r", encoding="utf-8") as f_in:
            prompts_data: list[dict[str, str]] = json.load(f_in)
            if not isinstance(prompts_data, list):
                raise ValueError(
                    "Input prompts file must contain a JSON array"
                )

        with open(config.function_definiton, "r", encoding="utf-8") as f_def:
            functions_data = json.load(f_def)
            if not isinstance(functions_data, list):
                raise ValueError(
                    "Function definitons file must contain a JSON array"
                )




if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
