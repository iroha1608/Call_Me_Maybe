import sys
import math

from src.tokenizer import Tokenizer
from src.models import FunctionDefinition


class ConstraintFilter:

    def __init__(
        self,
        tokenizer: Tokenizer,
        available_functions: list[FunctionDefinition]
    ) -> None:
        self._tokenizer = tokenizer
        self._available_functions = available_functions

    def filter_logits(
        self, logits: list[float], generated_text: str
    ) -> list[float]:
        try:
            filtered_logits = list(logits)
            return filtered_logits

        except Exception as e:
            print(f"ConstrainFilter: Error during logit filtering. "
                  f"{e}", file=sys.stderr)
            return logits
