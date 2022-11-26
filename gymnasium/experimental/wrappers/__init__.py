"""Experimental Wrappers."""
# isort: skip_file

from typing import TypeVar

ArgType = TypeVar("ArgType")

from gymnasium.experimental.wrappers.lambda_action import LambdaActionV0
from gymnasium.experimental.wrappers.lambda_observations import LambdaObservationsV0
from gymnasium.experimental.wrappers.lambda_reward import ClipRewardsV0, LambdaRewardV0

__all__ = [
    "ArgType",
    # Lambda Action
    "LambdaActionV0",
    # Lambda Observations
    "LambdaObservationsV0",
    # Lambda Rewards
    "LambdaRewardV0",
    "ClipRewardsV0",
]
