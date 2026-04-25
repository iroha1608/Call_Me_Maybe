import sys
import json
from src.llm_client import LLMClient


class TokenizerError(Exception):
    pass


class Tokenizer:

    def __init__(self, llm_client: LLMClient) -> None:
        self._vocab_path = llm_client.get_path_to_vocabfile()
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._load_vocablary()

    def _load_vocablary(self) -> None:
        try:
            with open(self._vocab_path, "r",  encoding="utf-8") as f:
                vocab_data = json.load(f)

            if not isinstance(vocab_data, dict):
                raise ValueError("Vocabulaty JSON is not a dictionary.")

            self._token_to_id = vocab_data
            self._id_to_token = {v: k for k, v in self._token_to_id.items()}

        except Exception as e:
            print(f"TokenizerError: Failed to load vocabulary."
                  f"{e}", file=sys.stderr)
            raise TokenizerError("Vocabulary loading failed.") from e

    def encode(self, text: str) -> list[int]:
        token_ids: list = []
        try:
            for char in text:
                if char in self._token_to_id:
                    token_ids.append(self._token_to_id[char])
                else:
                    pass
            return token_ids

        except Exception as e:
            print(f"TokenizerError: Encoding failed."
                  f"{e}", file=sys.stderr)
            raise TokenizerError("Encoding process failed.") from e

    def decode(self, token_ids: list[int]) -> str:
        try:
            tokens: list[str] = []
            for t_id in token_ids:
                if t_id in self._id_to_token:
                    token_str = self._id_to_token[t_id].replace("Ġ", " ")
                    tokens.append(token_str)
            return "".join(tokens)

        except Exception as e:
            print(f"TokenizerError: Decoding failed."
                  f"{e}", file=sys.stderr)
            raise TokenizerError("Decoding process failed.") from e
