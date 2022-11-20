"""Root __init__ of the gym dev_wrappers."""
from typing import Dict, Sequence, TypeVar, Union

ArgType = TypeVar("ArgType")

ParameterType = TypeVar("ParameterType")  # (float, float)
TreeParameterType = Union[
    ParameterType, Dict[str, "TreeParameterType"], Sequence["TreeParameterType"]
]
