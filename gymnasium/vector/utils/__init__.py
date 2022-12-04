"""Module for gymnasium vector utils."""
from gymnasium.vector.utils.misc import CloudpickleWrapper, clear_mpi_env_vars
from gymnasium.vector.utils.numpy_utils import concatenate, create_empty_array
from gymnasium.vector.utils.shared_memory import (
    create_shared_memory,
    read_from_shared_memory,
    write_to_shared_memory,
)
from gymnasium.vector.utils.spaces import (  # pyright: reportPrivateUsage=false
    BaseGymSpaces,
    _BaseGymSpaces,
    batch_space,
    iterate,
)


__all__ = [
    "CloudpickleWrapper",
    "clear_mpi_env_vars",
    "concatenate",
    "create_empty_array",
    "create_shared_memory",
    "read_from_shared_memory",
    "write_to_shared_memory",
    "BaseGymSpaces",
    "batch_space",
    "iterate",
]
