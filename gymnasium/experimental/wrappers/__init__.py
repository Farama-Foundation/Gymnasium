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


def __getattr__(wrapper_name):
    """Raises a DeprecatedWrapper exception when an old version of a wrapper is imported, or an ImportError if the wrapper being imported does not exist."""
    wrapper_module = __name__

    version = wrapper_name[-1]
    base = wrapper_name[:-1]

    try:
        version_num = int(version)
        is_valid_version = True
    except ValueError:
        is_valid_version = False

    global_dict = globals()

    if is_valid_version:
        for act_version_num in range(1000):
            act_wrapper_name = f"{base}{act_version_num}"
            if act_wrapper_name in global_dict:
                if version_num < act_version_num:
                    raise DeprecatedWrapper(
                        f"{base}{version_num} is now deprecated, use {act_wrapper_name} instead.\n"
                        f"To see the changes made, go to "
                        f"https://gymnasium.farama.org/api/experimental/wrappers/#{wrapper_module}.{act_wrapper_name}."
                    )

    raise ImportError(f"cannot import name '{wrapper_name}' from '{wrapper_module}'")
