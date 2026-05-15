"""
    LLM Client for interacting with the Small_LLM_Model.
    # type: ignore[attr-defined]
"""
from typing import cast
from llm_sdk import Small_LLM_Model


class LLMClientError(Exception):
    """Custom exception for errors that occur within the LLMClient."""
    pass


class LLMClient:

    def __init__(self) -> None:
        """
        Initialize the LLMClient by creating an instance of
        the Small_LLM_Model.
        Raises:
            LLMClientError: If the Small_LLM_Model fails to initialize.
        """
        try:
            self._model = Small_LLM_Model()
        except Exception as e:
            raise LLMClientError(
                "Failed to initialize Small_LLM_Model: {e}"
            ) from e

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """
            Retrieve logits for a given sequence of input token IDs.
        Args:
            input_ids (list[int]):
                A list of token IDs representing the input sequence.
        Returns:
            list[float]:
                A list of logits corresponding to the input token IDs.
        Raises:
            LLMClientError:
                If the input_ids list is empty or if there is an error
                while retrieving logits from the model.
        """
        if not input_ids:
            raise LLMClientError(
                "Cannot retrieve logits: input_ids is empty.")
        try:
            return self._model.get_logits_from_input_ids(input_ids)
        except Exception as e:
            raise LLMClientError(
                "Failed to retrieve logits for sequence "
                f"of length {len(input_ids)}. Underlying error: {e}"
            ) from e

    def encode(self, text: str) -> list[int]:
        """
            Encode a given text string into a list of token IDs.
        Args:
            text (str): The input text to be encoded.
        Returns:
            list[int]: A list of token IDs representing the encoded text.
        Raises:
            LLMClientError: If the input text is empty or if there is an error
            while encoding the text using the model.
        """
        if not text:
            return []
        try:
            encoded_tensor = self._model.encode(text)
            return cast(list[int], encoded_tensor[0].tolist())
        except Exception as e:
            raise LLMClientError("Failed to encode text.") from e

    def decode(self, token_ids: list[int]) -> str:
        """"
            Decode a list of token IDs back into a text string.
        Args:
            token_ids (list[int]): A list of token IDs to be decoded.
        Returns:
            str: The decoded text string corresponding to the input token IDs.
        Raises:
            LLMClientError: If the token_ids list is empty or
            if there is an error while decoding the token IDs using the model.
        """
        if not token_ids:
            return ""
        try:
            return self._model.decode(token_ids)
        except Exception as e:
            raise LLMClientError("Failed to decode token IDs.") from e

    def get_path_to_vocabfile(self) -> str:
        """"
            Retrieve the file path to the vocabulary file used by the model.
        Returns:
            str: The file path to the vocabulary file.
        Raises:
            LLMClientError: If there is an error
                while retrieving the vocab_file path from the model.
        """
        try:
            return self._model.get_path_to_vocab_file()
        except Exception as e:
            raise LLMClientError("Failed to get vocab_file path.") from e

    def get_path_to_mergefile(self) -> str:
        """
            Retrieve the file path to the merges file used by the model.
        Returns:
            str: The file path to the merges file.
        Raises:
            LLMClientError: If there is an error
                while retrieving the merges_file path from the model.
        """
        try:
            return self._model.get_path_to_merges_file()
        except Exception as e:
            raise LLMClientError("Failed to get merges_file path.") from e

    def get_path_to_tokenfile(self) -> str:
        """
            Retrieve the file path to the tokenizer file used by the model.
        Returns:
            str: The file path to the tokenizer file.
        Raises:
            LLMClientError: If there is an error
                while retrieving the tokenizer_file path from the model.
        """
        try:
            return self._model.get_path_to_tokenizer_file()
        except Exception as e:
            raise LLMClientError("Failed to get tokenizer_file path.") from e
