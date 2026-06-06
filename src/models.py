"""
    This module defines the data models for the Call Me Maybe engine,
    including the input prompt, function definitions,
    and the results of function calls.
    These models are used for validating and structuring the data
    that flows through the engine, ensuring that the inputs
    and outputs conform to the expected formats and types.
"""


from typing import Any
from pydantic import BaseModel, Field, field_validator


class PromptInput(BaseModel):
    """The input model for a natural language prompt."""
    prompt: str = Field(
        ...,
        description="The natural language prompt provided by the user."
    )


class ParameterDefinition(BaseModel):
    """The definition of a function parameter."""
    type: str = Field(
        ...,
        description="The data type of the parameter "
        "(e.g., 'string', 'number', 'boolean', 'null')."
    )

    @field_validator("type")
    def validator_supported_types(cls, v: str) -> str:
        """
            Validate that the provided type is supported by the engine.
            Args:
                v (str): The data type to validate.
            Returns:
                str: The validated data type in lowercase.
            Raises:
                ValueError: If the provided type is not supported.
        """
        supported_types = {
            "str", "string", "num", "number",
            "int", "integer", "bool", "boolean", "null"
        }
        v_lower = v.lower()
        if v_lower not in supported_types:
            raise ValueError(
                f"Unsupported parameter type {v}. "
                f"Engine currently supports: {supported_types}"
            )
        return v_lower


class FunctionDefinition(BaseModel):
    """The definition of a function that can be called by the engine."""
    name: str = Field(
        ...,
        description="The name of the function."
    )
    description: str = Field(
        default="No description provided.",
        description="A brief description of what the function does."
    )
    parameters: dict[str, ParameterDefinition] = Field(
        ...,
        description="A dictionary mapping parameter names "
        "to their definitions."
    )
    returns: ParameterDefinition | dict[str, Any] | None = Field(
        default=None,
        description="The return type definition of the function."
    )


class FunctionCallResult(BaseModel):
    """
        The result of a function call, including the original prompt,
        the name of the function called, and the extracted parameters.
    """
    prompt: str = Field(
        ...,
        description="The original natural-language request."
    )
    name: str = Field(
        ...,
        description="The name of function to call."
    )
    parameters: dict[str, Any] = Field(
        ...,
        description="The extracted arguments with their correct types."
    )
