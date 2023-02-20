"""Experimental Wrappers."""
# isort: skip_file

from gymnasium.experimental.wrappers.lambda_action import (
    LambdaActionV0,
    ClipActionV0,
    RescaleActionV0,
)
from gymnasium.experimental.wrappers.lambda_observations import (
    LambdaObservationV0,
    FilterObservationV0,
    FlattenObservationV0,
    GrayscaleObservationV0,
    ResizeObservationV0,
    ReshapeObservationV0,
    RescaleObservationV0,
    DtypeObservationV0,
    PixelObservationV0,
    NormalizeObservationV0,
)
from gymnasium.experimental.wrappers.lambda_reward import (
    ClipRewardV0,
    LambdaRewardV0,
    NormalizeRewardV0,
)
from gymnasium.experimental.wrappers.stateful_action import StickyActionV0
from gymnasium.experimental.wrappers.stateful_observation import (
    TimeAwareObservationV0,
    DelayObservationV0,
    FrameStackObservationV0,
)
from gymnasium.experimental.wrappers.atari_preprocessing import AtariPreprocessingV0
from gymnasium.experimental.wrappers.common import (
    PassiveEnvCheckerV0,
    OrderEnforcingV0,
    AutoresetV0,
    RecordEpisodeStatisticsV0,
)
from gymnasium.experimental.wrappers.rendering import (
    RenderCollectionV0,
    RecordVideoV0,
    HumanRenderingV0,
)

from gymnasium.experimental.wrappers.vector import (
    VectorRecordEpisodeStatistics,
    VectorListInfo,
)

__all__ = [
    # --- Observation wrappers ---
    "LambdaObservationV0",
    "FilterObservationV0",
    "FlattenObservationV0",
    "GrayscaleObservationV0",
    "ResizeObservationV0",
    "ReshapeObservationV0",
    "RescaleObservationV0",
    "DtypeObservationV0",
    "PixelObservationV0",
    "NormalizeObservationV0",
    "TimeAwareObservationV0",
    "FrameStackObservationV0",
    "DelayObservationV0",
    "AtariPreprocessingV0",
    # --- Action Wrappers ---
    "LambdaActionV0",
    "ClipActionV0",
    "RescaleActionV0",
    # "NanAction",
    "StickyActionV0",
    # --- Reward wrappers ---
    "LambdaRewardV0",
    "ClipRewardV0",
    "NormalizeRewardV0",
    # --- Common ---
    "AutoresetV0",
    "PassiveEnvCheckerV0",
    "OrderEnforcingV0",
    "RecordEpisodeStatisticsV0",
    # --- Rendering ---
    "RenderCollectionV0",
    "RecordVideoV0",
    "HumanRenderingV0",
    # --- Vector ---
    "VectorRecordEpisodeStatistics",
    "VectorListInfo",
]


class DeprecatedWrapper(ImportError):
    """Exception raised when an old version of a wrapper is imported."""

    pass


class InvalidVersionWrapper(ImportError):
    """Exception raised when an invalid version of a wrapper is imported."""

    pass


def __getattr__(wrapper_name):
    """Raise errors when importing deprecated or invalid versions of wrappers.

    Args:
        wrapper_name (str): The name of the wrapper being imported.

    Raises:
        DeprecatedWrapper: If the version is not the latest.
        InvalidVersionWrapper: If the version doesn't exist.
        ImportError: If the wrapper does not exist.
    """
    invalid_attrs = ("__wrapped__", "__test__")
    if wrapper_name in invalid_attrs:
        raise AttributeError(
            f"module 'gymnasium.experimental.wrappers' has no attribute {wrapper_name}"
        )

    base_name = wrapper_name[:-1]
    version = wrapper_name[-1]

    # Get all wrappers that start with the base wrapper name
    wrappers = [name for name in globals().keys() if name.startswith(base_name)]

    # If the wrapper does not exist, raise an ImportError
    if not wrappers:
        raise ImportError(
            f"cannot import name '{wrapper_name}' from 'gymnasium.experimental.wrappers'"
        )

    # Get the latest version of the wrapper
    latest_version = max([int(name[-1]) for name in wrappers])

    # Raise an InvalidVersionWrapper exception if the version is not a digit
    if not version.isdigit():
        raise InvalidVersionWrapper(
            f"{wrapper_name} is not a valid version number, use {base_name}{latest_version} instead."
        )

    version = int(version)

    # Raise a DeprecatedWrapper exception if the version is not the latest
    if version < latest_version:
        raise DeprecatedWrapper(
            f"{wrapper_name} is now deprecated, use {base_name}{latest_version} instead.\n"
            f"To see the changes made, go to "
            f"https://gymnasium.farama.org/api/experimental/wrappers/#gymnasium.experimental.wrappers.{base_name}{latest_version}."
        )
    # Raise an InvalidVersionWrapper exception if the version is greater than the latest
    elif version > latest_version:
        raise InvalidVersionWrapper(
            f"{wrapper_name} is the wrong version number, use {base_name}{latest_version} instead."
        )
