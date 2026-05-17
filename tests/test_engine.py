"""Unit tests for the GenerationEngine class in src.engine."""
import pytest
from unittest.mock import MagicMock
from src.engine import GenerationEngine, EngineError


class TestGenerationEngine:
    """Unit tests for the GenerationEngine class."""
    @pytest.fixture
    def mock_components(self) -> tuple[MagicMock, MagicMock, MagicMock]:
        """
            Fixture to create mock components for the GenerationEngine.
            Returns:
                A tuple containing mock objects for the LLM,
                tokenizer, and filter.
        """
        llm_mock = MagicMock()
        tokenizer_mock = MagicMock()
        filter_mock = MagicMock()
        return llm_mock, tokenizer_mock, filter_mock

    def test_generate_success(self, mock_components: MagicMock) -> None:
        """
            Test the successful generation of a response
            by the GenerationEngine.
            This test simulates a scenario where the engine generates
            a valid JSON response within the maximum token limit.
            The test verifies that the generated response is correctly parsed
            and returned as a dictionary.
            It also checks that the tokenizer's decode method is called
            the expected number of times.
            Args:
                mock_components: A fixture that provides mock objects
                for the LLM, tokenizer, and filter.
        """
        llm, tokenizer, filter_mock = mock_components

        tokenizer.encode.return_value = [1, 2, 3]
        llm.get_logits.return_value = [0.1, 0.9, 0.2]
        filter_mock.filter_logits.return_value = [0.1, 0.9, 0.2]

        def mock_decode(t_ids: list[int]) -> str:
            if len(t_ids) == 1:
                return "{"
            elif len(t_ids) == 2:
                return '{"a": 1'
            elif len(t_ids) == 3:
                return '{"name": "test", "parameters": {}}'
            return ""

        tokenizer.decode.side_effect = mock_decode
        engine = GenerationEngine(
            llm, tokenizer, filter_mock, max_new_tokens=10
        )
        result = engine.generate("dummy prompt")

        assert isinstance(result, dict)
        assert result["name"] == "test"

    def test_generate_max_tokens_reached(
        self, mock_components: MagicMock
    ) -> None:
        """
            Test the behavior of the GenerationEngine when the maximum token
            limit is reached during generation.
            This test simulates a scenario where the engine generates tokens
            but fails to produce a valid JSON response within the specified
            maximum token limit. The test verifies that an EngineError is
            raised with the appropriate error message when
            the limit is exceeded.
            Args:
                mock_components: A fixture that provides mock objects
                for the LLM, tokenizer, and filter.
        """
        llm, tokenizer, filter_mock = mock_components

        tokenizer.encode.return_value = [1]
        llm.get_logits.return_value = [0.5, 0.5]
        filter_mock.filter_logits.return_value = [0.5, 0.5]

        tokenizer.decode.return_value = '{"name": "test"'

        engine = GenerationEngine(
            llm, tokenizer, filter_mock, max_new_tokens=3
        )

        with pytest.raises(EngineError, match="Maximum tokens"):
            engine.generate("dummy prompt")
