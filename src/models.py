from typing import Any
from pydantic import BaseModel, Field


class PromptInput(BaseModel):
    prompt: str = Field(
        ...,
        description="The natural language prompt provided by the user."
    )


class ParameterDefinition(BaseModel):
    type: str = Field(
        ...,
        description="The data type of the parameter "
        "(e.g., 'string', 'number')."
    )


class FunctionDefinition(BaseModel):
    name: str = Field(
        ...,
        description="The name of the function."
    )
    description: str = Field(
        ...,
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
        description="The original natural-langage request."
    )
    name: str = Field(
        ...,
        description="The name of function to call."
    )
    parameters: dict[str, Any] = Field(
        ...,
        description="The extracted arguments with their correct types."
    )
