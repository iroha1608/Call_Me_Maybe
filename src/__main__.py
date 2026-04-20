import sys

from src.cli import parse_arguments
from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints import ConstrainFilter
from src.engine import GenerationEngine
from src.models import FunctionCallResult


def main() -> None:




if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
