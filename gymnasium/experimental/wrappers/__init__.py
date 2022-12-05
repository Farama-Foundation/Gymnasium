"""Experimental Wrappers."""
# isort: skip_file

from typing import TypeVar

ArgType = TypeVar("ArgType")

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
)
from gymnasium.experimental.wrappers.lambda_reward import ClipRewardV0, LambdaRewardV0
from gymnasium.experimental.wrappers.numpy_to_jax import JaxToNumpyV0
from gymnasium.experimental.wrappers.torch_to_jax import JaxToTorchV0
from gymnasium.experimental.wrappers.stateful_action import StickyActionV0
from gymnasium.experimental.wrappers.stateful_observation import (
    TimeAwareObservationV0,
    DelayObservationV0,
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
    # "PixelObservationV0",
    # "NormalizeObservationV0",
    "TimeAwareObservationV0",
    # "FrameStackV0",
    "DelayObservationV0",
    # "AtariPreprocessingV0"
    # --- Action Wrappers ---
    "LambdaActionV0",
    "ClipActionV0",
    "RescaleActionV0",
    # "NanAction",
    "StickyActionV0",
    # --- Reward wrappers ---
    "LambdaRewardV0",
    "ClipRewardV0",
    # "RescaleRewardV0",
    # "NormalizeRewardV0",
    # --- Common ---
    # "AutoReset",
    # "PassiveEnvChecker",
    # "OrderEnforcing",
    # "RecordEpisodeStatistics",
    # "RenderCollection",
    # "HumanRendering",
    "JaxToNumpyV0",
    "JaxToTorchV0",
]
