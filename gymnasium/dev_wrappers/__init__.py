"""Root __init__ of the gym dev_wrappers."""
from typing import Dict, Sequence, TypeVar, Union

ArgType = TypeVar("ArgType")

# type parameters for handling arbitrarily nested spaces
ParameterType = TypeVar("ParameterType")
CompositeParameterType = Union[
    ParameterType,
    Dict[str, "CompositeParameterType"],
    Sequence["CompositeParameterType"],
]
