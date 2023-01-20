"""A collection of wrappers for modifying the reward.

* ``LambdaReward`` - Transforms the reward by a function
* ``ClipReward`` - Clips the reward between a minimum and maximum value
"""

from __future__ import annotations

from typing import Any, Callable, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.error import InvalidBound
from gymnasium.experimental.wrappers.utils import RunningMeanStd


class LambdaRewardV0(gym.RewardWrapper):
    """A reward wrapper that allows a custom function to modify the step reward.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import LambdaRewardV0
        >>> env = gym.make("CartPole-v1")
        >>> env = LambdaRewardV0(env, lambda r: 2 * r + 1)
        >>> _ = env.reset()
        >>> _, rew, _, _, _ = env.step(0)
        >>> rew
        3.0
    """

    def __init__(
        self,
        env: gym.Env,
        func: Callable[[SupportsFloat], SupportsFloat],
    ):
        """Initialize LambdaRewardV0 wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            func: (Callable): The function to apply to reward
        """
        super().__init__(env)

        self.func = func

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Apply function to reward.

        Args:
            reward (Union[float, int, np.ndarray]): environment's reward
        """
        return self.func(reward)


class ClipRewardV0(LambdaRewardV0):
    """A wrapper that clips the rewards for an environment between an upper and lower bound.

    Example with an upper and lower bound:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import ClipRewardV0
        >>> env = gym.make("CartPole-v1")
        >>> env = ClipRewardV0(env, 0, 0.5)
        >>> _ = env.reset()
        >>> _, rew, _, _, _ = env.step(1)
        >>> rew
        0.5
    """

    def __init__(
        self,
        env: gym.Env,
        min_reward: float | np.ndarray | None = None,
        max_reward: float | np.ndarray | None = None,
    ):
        """Initialize ClipRewardsV0 wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_reward (Union[float, np.ndarray]): lower bound to apply
            max_reward (Union[float, np.ndarray]): higher bound to apply
        """
        if min_reward is None and max_reward is None:
            raise InvalidBound("Both `min_reward` and `max_reward` cannot be None")

        elif max_reward is not None and min_reward is not None:
            if np.any(max_reward - min_reward < 0):
                raise InvalidBound(
                    f"Min reward ({min_reward}) must be smaller than max reward ({max_reward})"
                )

        super().__init__(env, lambda x: np.clip(x, a_min=min_reward, a_max=max_reward))


class NormalizeRewardV0(gym.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the reward
    statistics. If `True` (default), the `RunningMeanStd` will get updated every time `self.normalize()` is called.
    If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.rewards_running_means = RunningMeanStd(shape=())
        self.discounted_reward: np.array = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(action)
        self.discounted_reward = self.discounted_reward * self.gamma * (
            1 - terminated
        ) + float(reward)
        return obs, self.normalize(float(reward)), terminated, truncated, info

    def normalize(self, reward):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.rewards_running_means.update(self.discounted_reward)
        return reward / np.sqrt(self.rewards_running_means.var + self.epsilon)
