import json
from src.llm_client import LLMClient, LLMClientError


class TokenizerError(Exception):
    """Tokenizer、語彙処理に関する独自例外"""
    pass


class Tokenizer:

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

        try:
            self._vocab_path = llm_client.get_path_to_vocabfile()
        except LLMClientError as e:
            raise TokenizerError(
                f"Failed to retrieve vocab path from LLMClient: {e}"
            ) from e

        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        try:
            with open(self._vocab_path, "r",  encoding="utf-8") as f:
                vocab_data = json.load(f)

            if not isinstance(vocab_data, dict):
                raise ValueError("Vocabulaty JSON is not a dictionary.")

            self._token_to_id = vocab_data
            self._id_to_token = {v: k for k, v in self._token_to_id.items()}

        except FileNotFoundError as e:

            raise TokenizerError(
                f"Vocabulary file not found at {self._vocab_path}. "
                "This model might use a different tokenizer format "
                "(e.g., SentencePiece) Which lacks a standard vocab.json"
            ) from e

        except json.JSONDecodeError as e:
            raise TokenizerError(
                f"Vocabulary file is corrupted or invalid JSON: {e}"
            ) from e

        except Exception as e:
            raise TokenizerError("Vocabulary loading failed.") from e

    def encode(self, prompt: str) -> list[int]:
        try:
            return self._llm_client.encode(prompt)

        except LLMClientError as e:
            raise TokenizerError("Encoding process failed.") from e

    def decode(self, token_ids: list[int]) -> str:
        try:
            return self._llm_client.decode(token_ids)

        except LLMClientError as e:
            raise TokenizerError("Decoding process failed.") from e
