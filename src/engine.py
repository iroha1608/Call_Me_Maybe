import os
import sys
import json
from typing Any

from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints import ConstrainFilter


class EngineError:
    pass


class GenerationEngine:

    def __init__(
        self,
        llm_client: LLMClient,
        tokenizer: Tokenizer,
        constraint_filter: ConstraintsFilter,
        max_new_tokens: int=256
        ) -> None:
        self._llm = llm_client
        self._tokenizer = tokenizer
        self._constraint_filter = constraint_filter
        self._max_new_tokens = max_new_tokens

    def _argmax(self, logits: list[float]) -> int:
        if not logits:
            raise ValueError("Empty logits list provided to argmax.")
        return logits.index(max(logits))

    def generate(self, prompt: str) -> dict[str, Any]:
        input_ids = self._tokenizer.encode(prompt)
        generate_ids: list[int] = []
        generated_text = ""

        for _ in range(self._max_tokens):
            current_sequence = input_ids + generated_ids
            logits = self._llm.get_logits(current_sequence)

            filtered_logits = self._filter.filter_logits(
                logits=logits,
                generated_text=generated_text
            )
            next_token_id = self._argmax(filtered_logits)
            generated_ids.append(next_token_id)

            generated_text = self._tokenizer.decode(generated_ids)

            if generated_text.endswith("}"):
                try:
                    parsed_json = json.loads(generated_text)
                    if isinstance(parsed_json, dict):
                        return parsed_json
                except json.JSONDecodeError:
                    pass
            raise EngineError(
                "Maximum token reached without generating valid JSON."
            )

        except Exception as e:
            print(f"EngineError: Generation pipeline failed."
                  f"{e}", file=sys.stderr)
            return {"error": str(e), "status": "failed"}
