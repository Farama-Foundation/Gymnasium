"""Experimental Wrappers."""
# isort: skip_file

from typing import TypeVar

ArgType = TypeVar("ArgType")

from gymnasium.experimental.wrappers.lambda_action import LambdaActionV0
from gymnasium.experimental.wrappers.lambda_observations import LambdaObservationV0
from gymnasium.experimental.wrappers.lambda_reward import ClipRewardV0, LambdaRewardV0

__all__ = [
    "ArgType",
    # Lambda Action
    "LambdaActionV0",
    # Lambda Observation
    "LambdaObservationV0",
    # Lambda Reward
    "LambdaRewardV0",
    "ClipRewardV0",
]
