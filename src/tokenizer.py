"""
    Tokenizer module that provides functionality to encode
    and decode text using a vocabulary file.
    The Tokenizer class interfaces with the LLMClient
    to perform encoding and decoding operations,
    while also managing the vocabulary loaded from a JSON file.
"""
import json
from src.llm_client import LLMClient, LLMClientError


class TokenizerError(Exception):
    """Custom exception for errors that occur during tokenization processes."""
    pass


class Tokenizer:
    """
        Tokenizer that interfaces with the LLMClient to encode and decode text.
        It also loads the vocabulary from a JSON file provided
        by the LLMClient.
    """
    def __init__(self, llm_client: LLMClient) -> None:
        """
            Initialize the Tokenizer
                    with the given LLMClient and load the vocabulary.
            Args:
                llm_client (LLMClient):
                    The language model client to interface with.
            Raises:
                TokenizerError:
                    If there is an error while loading the vocabulary.
        """
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
        """
            Loads the vocabulary from the JSON file.
            The vocabulary file is expected to be a JSON object
            where keys are tokens (strings)
            and values are their corresponding token IDs (integers).
            Raises:
                TokenizerError:
                    If the vocabulary file is not found, is not a valid JSON,
                    or if there is any error during the loading process.
        """
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
        """
            Encodes a string into a list of token IDs.
            Args:
                prompt (str): The input string to encode.
            Returns:
                list[int]: The list of token IDs.
            Raises:
                TokenizerError: If the encoding process fails.
        """
        try:
            return self._llm_client.encode(prompt)

        except LLMClientError as e:
            raise TokenizerError("Encoding process failed.") from e

    def decode(self, token_ids: list[int]) -> str:
        """
            Decodes a list of token IDs back into a string.
            Args:
                token_ids (list[int]): The list of token IDs to decode.
            Returns:
                str: The decoded string.
            Raises:
                TokenizerError: If the decoding process fails.
        """
        try:
            return self._llm_client.decode(token_ids)

        except LLMClientError as e:
            raise TokenizerError("Decoding process failed.") from e
