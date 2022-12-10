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
    PixelObservationV0,
    NormalizeObservationV0,
)
from gymnasium.experimental.wrappers.lambda_reward import (
    ClipRewardV0,
    LambdaRewardV0,
    NormalizeRewardV0,
)
from gymnasium.experimental.wrappers.jax_to_numpy import JaxToNumpyV0
from gymnasium.experimental.wrappers.jax_to_torch import JaxToTorchV0
from gymnasium.experimental.wrappers.numpy_to_torch import NumpyToTorchV0
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
    # --- Data Conversion ---
    "JaxToNumpyV0",
    "JaxToTorchV0",
    "NumpyToTorchV0",
]
