import sys
from typing import cast
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class LLMClientError(Exception):
    pass


class LLMClient:

    def __init__(self) -> None:
        try:
            self._model = Small_LLM_Model()
        except Exception as e:
            print(f"LLMClientError: Initialization failed."
                  f"{e}", file=sys.stderr)
            raise LLMClientError(
                "Failed to initialize Small_LLM_Model."
            ) from e

    def get_logits(self, input_ids: list[int]) -> list[float]:
        try:
            return cast(
                list[float], self._model.get_logits_from_input_ids(input_ids)
            )
        except Exception as e:
            print(f"LLMClientError: Logit retrieval failed."
                  f"{e}", file=sys.stderr)
            raise LLMClientError("Failed to retrieve logits.") from e

    def get_path_to_vocabfile(self) -> str:
        try:
            return cast(str, self._model.get_path_to_vocab_file())
        except Exception as e:
            print(f"LLMClientError: Vocab_file path retrieval failed."
                  f"{e}", file=sys.stderr)
            raise LLMClientError("Failed to get vocab_file path.") from e

    def get_path_to_mergefile(self) -> str:
        try:
            return cast(str, self._model.get_path_to_merges_file())
        except Exception as e:
            print(f"LLMClientError: Merges_file path retieval failed."
                  f"{e}", file=sys.stderr)
            raise LLMClientError("Failed to  get merges_file path.") from e

    def get_path_to_tokenfile(self) -> str:
        try:
            return cast(str, self._model.get_path_to_tokenizer_file())
        except Exception as e:
            print(f"LLMClientError: Tokenizer_file path retieval failed."
                  f"{e}", file=sys.stderr)
            raise LLMClientError("Failed to get tokenizer_file path.") from e
