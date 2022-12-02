"""Experimental Wrappers."""
# isort: skip_file

from typing import TypeVar

ArgType = TypeVar("ArgType")

from gymnasium.experimental.wrappers.lambda_action import (
    LambdaActionV0,
    ClipActionV0,
    RescaleActionV0,
)
from gymnasium.experimental.wrappers.lambda_observations import LambdaObservationV0
from gymnasium.experimental.wrappers.lambda_reward import ClipRewardV0, LambdaRewardV0
from gymnasium.experimental.wrappers.sticky_action import StickyActionV0
from gymnasium.experimental.wrappers.time_aware_observation import (
    TimeAwareObservationV0,
)
from gymnasium.experimental.wrappers.delay_observation import DelayObservationV0

__all__ = [
    "ArgType",
    # Lambda Action
    "LambdaActionV0",
    "StickyActionV0",
    "ClipActionV0",
    "RescaleActionV0",
    # Lambda Observation
    "LambdaObservationV0",
    "DelayObservationV0",
    "TimeAwareObservationV0",
    # Lambda Reward
    "LambdaRewardV0",
    "ClipRewardV0",
]
