"""Wrappers for vector environments."""
# pyright: reportUnsupportedDunderAll=false
import importlib
import re

from gymnasium.error import DeprecatedWrapper
from gymnasium.experimental.wrappers.vector.dict_info_to_list import DictInfoToListV0
from gymnasium.experimental.wrappers.vector.record_episode_statistics import (
    RecordEpisodeStatisticsV0,
)
from gymnasium.experimental.wrappers.vector.vectorize_action import (
    ClipActionV0,
    LambdaActionV0,
    RescaleActionV0,
    VectorizeLambdaActionV0,
)
from gymnasium.experimental.wrappers.vector.vectorize_observation import (
    DtypeObservationV0,
    FilterObservationV0,
    FlattenObservationV0,
    GrayscaleObservationV0,
    LambdaObservationV0,
    RescaleObservationV0,
    ReshapeObservationV0,
    ResizeObservationV0,
    VectorizeLambdaObservationV0,
)
from gymnasium.experimental.wrappers.vector.vectorize_reward import (
    ClipRewardV0,
    LambdaRewardV0,
    VectorizeLambdaRewardV0,
)


__all__ = [
    # --- Vector only wrappers
    "VectorizeLambdaObservationV0",
    "VectorizeLambdaActionV0",
    "VectorizeLambdaRewardV0",
    "DictInfoToListV0",
    # --- Observation wrappers ---
    "LambdaObservationV0",
    "FilterObservationV0",
    "FlattenObservationV0",
    "GrayscaleObservationV0",
    "ResizeObservationV0",
    "ReshapeObservationV0",
    "RescaleObservationV0",
    "DtypeObservationV0",
    # "PixelObservationV0",
    # "NormalizeObservationV0",
    # "TimeAwareObservationV0",
    # "FrameStackObservationV0",
    # "DelayObservationV0",
    # --- Action Wrappers ---
    "LambdaActionV0",
    "ClipActionV0",
    "RescaleActionV0",
    # --- Reward wrappers ---
    "LambdaRewardV0",
    "ClipRewardV0",
    # "NormalizeRewardV1",
    # --- Common ---
    "RecordEpisodeStatisticsV0",
    # --- Rendering ---
    # "RenderCollectionV0",
    # "RecordVideoV0",
    # "HumanRenderingV0",
    # --- Conversion ---
    "JaxToNumpyV0",
    "JaxToTorchV0",
    "NumpyToTorchV0",
]


# As these wrappers requires `jax` or `torch`, they are loaded by runtime on users trying to access them
#   to avoid `import jax` or `import torch` on `import gymnasium`.
_wrapper_to_class = {
    # data converters
    "JaxToNumpyV0": "jax_to_numpy",
    "JaxToTorchV0": "jax_to_torch",
    "NumpyToTorchV0": "numpy_to_torch",
}


def __getattr__(wrapper_name: str):
    """Load a wrapper by name.

    This optimizes the loading of gymnasium wrappers by only loading the wrapper if it is used.
    Errors will be raised if the wrapper does not exist or if the version is not the latest.

    Args:
        wrapper_name: The name of a wrapper to load.

    Returns:
        The specified wrapper.

    Raises:
        AttributeError: If the wrapper does not exist.
        DeprecatedWrapper: If the version is not the latest.
    """
    # Check if the requested wrapper is in the _wrapper_to_class dictionary
    if wrapper_name in _wrapper_to_class:
        import_stmt = (
            f"gymnasium.experimental.wrappers.vector.{_wrapper_to_class[wrapper_name]}"
        )
        module = importlib.import_module(import_stmt)
        return getattr(module, wrapper_name)

    # Define a regex pattern to match the integer suffix (version number) of the wrapper
    int_suffix_pattern = r"(\d+)$"
    version_match = re.search(int_suffix_pattern, wrapper_name)

    # If a version number is found, extract it and the base wrapper name
    if version_match:
        version = int(version_match.group())
        base_name = wrapper_name[: -len(version_match.group())]
    else:
        version = float("inf")
        base_name = wrapper_name[:-2]

    # Filter the list of all wrappers to include only those with the same base name
    matching_wrappers = [name for name in __all__ if name.startswith(base_name)]

    # If no matching wrappers are found, raise an AttributeError
    if not matching_wrappers:
        raise AttributeError(f"module {__name__!r} has no attribute {wrapper_name!r}")

    # Find the latest version of the matching wrappers
    latest_wrapper = max(
        matching_wrappers, key=lambda s: int(re.findall(int_suffix_pattern, s)[0])
    )
    latest_version = int(re.findall(int_suffix_pattern, latest_wrapper)[0])

    # If the requested wrapper is an older version, raise a DeprecatedWrapper exception
    if version < latest_version:
        raise DeprecatedWrapper(
            f"{wrapper_name!r} is now deprecated, use {latest_wrapper!r} instead.\n"
            f"To see the changes made, go to "
            f"https://gymnasium.farama.org/api/experimental/vector-wrappers/#gymnasium.experimental.wrappers.vector.{latest_wrapper}"
        )
    # If the requested version is invalid, raise an AttributeError
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {wrapper_name!r}, did you mean {latest_wrapper!r}"
        )
