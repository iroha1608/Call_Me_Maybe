from typing import cast
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class LLMClientError(Exception):
    pass


class LLMClient:

    def __init__(self) -> None:
        try:
            self._model = Small_LLM_Model()
        except Exception as e:
            raise LLMClientError(
                "Failed to initialize Small_LLM_Model: {e}"
            ) from e

    def get_logits(self, input_ids: list[int]) -> list[float]:
        if not input_ids:
            raise LLMClientError(
                "Cannot retrieve logits: input_ids is empty.") from e
        try:
            return cast(
                list[float], self._model.get_logits_from_input_ids(input_ids)
            )
        except Exception as e:
            raise LLMClientError(
                "Failed to retrieve logits for sequence "
                f"of length {len(input_ids)}. Underlying error: {e}"
            ) from e

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        try:
            encoded_tensor = self._model.encode(text)
            return cast(list[int], encoded_tensor[0].tolist())
        except Exception as e:
            raise LLMClientError("Failed to encode text.") from e

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            return cast(str, self._model.decode(token_ids))
        except Exception as e:
            raise LLMClientError("Failed to decode token IDs.") from e

    def get_path_to_vocabfile(self) -> str:
        try:
            return cast(str, self._model.get_path_to_vocab_file())
        except Exception as e:
            raise LLMClientError("Failed to get vocab_file path.") from e

    def get_path_to_mergefile(self) -> str:
        try:
            return cast(str, self._model.get_path_to_merges_file())
        except Exception as e:
            raise LLMClientError("Failed to get merges_file path.") from e

    def get_path_to_tokenfile(self) -> str:
        try:
            return cast(str, self._model.get_path_to_tokenizer_file())
        except Exception as e:
            raise LLMClientError("Failed to get tokenizer_file path.") from e
