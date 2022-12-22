from dataclasses import dataclass
from typing import Any


# todo: move this elsewhere
@dataclass
class WrapperSpec:
    """A specification for recording wrapper configs.

    * name: The name of the wrapper.
    * entry_point: The location of the wrapper to create from.
    * kwargs: Additional keyword arguments passed to the wrapper.
    """

    name: str
    entry_point: str
    kwargs: "list[Any]"
