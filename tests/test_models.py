"""Tests for the data models defined in src.models."""
import pytest
from pydantic import ValidationError
from src.models import ParameterDefinition, FunctionCallResult


class TestModels:
    """Unit tests for the data models defined in src.models."""
    def test_parameter_definition_valid_types(self) -> None:
        """
            Test that ParameterDefinition accepts valid types.
        """
        valid_types = ["string", "number", "boolean", "null", "integer"]
        for t in valid_types:
            param = ParameterDefinition(type=t)
            assert param.type == t.lower()

    def test_parameter_definition_invalid_type(self) -> None:
        """
            Test that ParameterDefinition raises a ValidationError for
            unsupported types.
        """
        with pytest.raises(
                ValidationError, match="Unsupported parameter type"):
            ParameterDefinition(type="array")

        with pytest.raises(
                ValidationError, match="Unsupported parameter type"):
            ParameterDefinition(type="object")

    def test_function_call_result_missing_keys(self) -> None:
        """
            Test that FunctionCallResult raises a ValidationError when
            required keys are missing.
        """
        with pytest.raises(ValidationError):
            FunctionCallResult(prompt="test", parameters={})
