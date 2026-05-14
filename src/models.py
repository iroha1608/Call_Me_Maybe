from typing import Any
from pydantic import BaseModel, Field, field_validator


class PromptInput(BaseModel):
    prompt: str = Field(
        ...,
        description="The natural language prompt provided by the user."
    )


class ParameterDefinition(BaseModel):
    type: str = Field(
        ...,
        description="The data type of the parameter "
        "(e.g., 'string', 'number', 'boolean', 'null')."
    )

    @field_validator("type")
    def validator_supported_types(cls, v: str) -> str:
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
    name: str = Field(
        ...,
        description="The name of the function."
    )
    description: str = Field(
        default="No description provided.",
        description="A brief description of what the function does."
    )
    parameters: dict[str, ParameterDefinition] = Field(
        default_factory=dict,
        description="A dictionary mapping parameter names "
        "to their definitions."
    )
    returns: ParameterDefinition | dict[str, Any] | None = Field(
        default=None,
        description="The return type definition of the function."
    )


class FunctionCallResult(BaseModel):
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
