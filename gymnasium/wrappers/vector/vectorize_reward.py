"""Vectorizes reward function to work with `VectorEnv`."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
from typing_extensions import TypeVar

from gymnasium import Env
from gymnasium.typing import (
    VectorActType,
    VectorBoolType,
    VectorObsType,
    VectorRewardType,
)
from gymnasium.vector import VectorEnv, VectorRewardWrapper
from gymnasium.wrappers import transform_reward

if TYPE_CHECKING:
    from typing import Protocol, type_check_only

    @type_check_only
    class _CanIterAndSetItem(Protocol):
        def __iter__(self) -> Any: ...
        def __setitem__(self, key: int, value: Any) -> None: ...


MutableRewardArrayType = TypeVar(
    "MutableRewardArrayType", bound="_CanIterAndSetItem", default=Any
)
# The wrapped (inner) environment's reward array type; defaults to this wrapper's own
# invariant `RewardArrayType`.
WrappedRewardArrayType = TypeVar("WrappedRewardArrayType", default=VectorRewardType)


class TransformReward(
    VectorRewardWrapper[
        VectorObsType,
        VectorActType,
        VectorRewardType,
        VectorBoolType,
        WrappedRewardArrayType,
    ],
    Generic[
        VectorObsType,
        VectorActType,
        VectorRewardType,
        VectorBoolType,
        WrappedRewardArrayType,
    ],
):
    """A reward wrapper that allows a custom function to modify the step reward.

    Example with reward transformation:
        >>> import gymnasium as gym
        >>> from gymnasium.spaces import Box
        >>> def scale_and_shift(rew):
        ...     return (rew - 1.0) * 2.0
        ...
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> envs = TransformReward(env=envs, func=scale_and_shift)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
        >>> envs.close()
        >>> obs
        array([[-4.6343064e-01,  9.8971417e-05],
               [-4.4488689e-01, -1.9375233e-03],
               [-4.3118435e-01, -1.5342437e-03]], dtype=float32)
    """

    func: Callable[[WrappedRewardArrayType], VectorRewardType]

    def __init__(
        self,
        env: VectorEnv[
            VectorObsType, VectorActType, WrappedRewardArrayType, VectorBoolType
        ],
        func: Callable[[WrappedRewardArrayType], VectorRewardType],
    ) -> None:
        """Initialize LambdaReward wrapper.

        Args:
            env (Env): The vector environment to wrap
            func: (Callable): The function to apply to reward
        """
        super().__init__(env)

        self.func = func

    def rewards(self, rewards: WrappedRewardArrayType) -> VectorRewardType:
        """Apply function to reward."""
        return self.func(rewards)


class VectorizeTransformReward(
    VectorRewardWrapper[
        VectorObsType,
        VectorActType,
        MutableRewardArrayType,
        VectorBoolType,
        MutableRewardArrayType,
    ],
    Generic[VectorObsType, VectorActType, MutableRewardArrayType, VectorBoolType],
):
    """Vectorizes a single-agent transform reward wrapper for vector environments.

    An example such that applies a ReLU to the reward:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformReward
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> envs = VectorizeTransformReward(envs, wrapper=TransformReward, func=lambda x: (x > 0.0) * x)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
        >>> envs.close()
        >>> rew
        array([-0., -0., -0.])
    """

    wrapper: transform_reward.TransformReward[Any, Any]

    def __init__(
        self,
        env: VectorEnv[
            VectorObsType, VectorActType, MutableRewardArrayType, VectorBoolType
        ],
        wrapper: type[transform_reward.TransformReward[Any, Any]],
        **kwargs: Any,
    ) -> None:
        """Constructor for the vectorized lambda reward wrapper.

        Args:
            env: The vector environment to wrap.
            wrapper: The wrapper to vectorize
            **kwargs: Keyword argument for the wrapper
        """
        super().__init__(env)

        self.wrapper = wrapper(Env(), **kwargs)

    def rewards(self, rewards: MutableRewardArrayType) -> MutableRewardArrayType:
        """Iterates over the reward updating each with the wrapper func."""
        for i, r in enumerate(rewards):
            rewards[i] = self.wrapper.func(r)
        return rewards


class ClipReward(
    VectorizeTransformReward[
        VectorObsType, VectorActType, MutableRewardArrayType, VectorBoolType
    ],
    Generic[VectorObsType, VectorActType, MutableRewardArrayType, VectorBoolType],
):
    """A wrapper that clips the rewards for an environment between an upper and lower bound.

    Example with clipped rewards:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> envs = ClipReward(envs, 0.0, 2.0)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> for _ in range(10):
        ...     obs, rew, term, trunc, info = envs.step(0.5 * np.ones((3, 1)))
        ...
        >>> envs.close()
        >>> rew
        array([0., 0., 0.])
    """

    def __init__(
        self,
        env: VectorEnv[
            VectorObsType, VectorActType, MutableRewardArrayType, VectorBoolType
        ],
        min_reward: float | np.ndarray | None = None,
        max_reward: float | np.ndarray | None = None,
    ) -> None:
        """Constructor for ClipReward wrapper.

        Args:
            env: The vector environment to wrap
            min_reward: The min reward for each step
            max_reward: the max reward for each step
        """
        super().__init__(
            env,
            transform_reward.ClipReward,
            min_reward=min_reward,
            max_reward=max_reward,
        )
