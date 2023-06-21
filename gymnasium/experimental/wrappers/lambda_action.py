"""A collection of wrappers that all use the LambdaAction class.

* ``LambdaActionV0`` - Transforms the actions based on a function
* ``ClipActionV0`` - Clips the action within a bounds
* ``RescaleActionV0`` - Rescales the action within a minimum and maximum actions
"""
from __future__ import annotations

from typing import Callable

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperActType
from gymnasium.spaces import Box, Space


__all__ = ["LambdaActionV0", "ClipActionV0", "RescaleActionV0"]


class LambdaActionV0(
    gym.ActionWrapper[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """A wrapper that provides a function to modify the action passed to :meth:`step`."""

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[WrapperActType], ActType],
        action_space: Space[WrapperActType] | None,
    ):
        """Initialize LambdaAction.

        Args:
            env: The environment to wrap
            func: Function to apply to the :meth:`step`'s ``action``
            action_space: The updated action space of the wrapper given the function.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, func=func, action_space=action_space
        )
        gym.Wrapper.__init__(self, env)

        if action_space is not None:
            self.action_space = action_space

        self.func = func

    def action(self, action: WrapperActType) -> ActType:
        """Apply function to action."""
        return self.func(action)


class ClipActionV0(
    LambdaActionV0[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import ClipActionV0
        >>> import numpy as np
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = ClipActionV0(env)
        >>> env.action_space
        Box(-inf, inf, (3,), float32)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([5.0, -2.0, 0.0], dtype=np.float32))
        ... # Executes the action np.array([1.0, -1.0, 0]) in the base environment
    """

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to wrap
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        LambdaActionV0.__init__(
            self,
            env=env,
            func=lambda action: np.clip(
                action, env.action_space.low, env.action_space.high
            ),
            action_space=Box(
                -np.inf,
                np.inf,
                shape=env.action_space.shape,
                dtype=env.action_space.dtype,
            ),
        )


class RescaleActionV0(
    LambdaActionV0[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import RescaleActionV0
        >>> import numpy as np
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> _ = env.reset(seed=42)
        >>> obs, _, _, _, _ = env.step(np.array([1, 1, 1], dtype=np.float32))
        >>> _ = env.reset(seed=42)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 0.75], dtype=np.float32)
        >>> wrapped_env = RescaleActionV0(env, min_action=min_action, max_action=max_action)
        >>> wrapped_env_obs, _, _, _, _ = wrapped_env.step(max_action)
        >>> np.alltrue(obs == wrapped_env_obs)
        True
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        min_action: float | int | np.ndarray,
        max_action: float | int | np.ndarray,
    ):
        """Constructor for the Rescale Action wrapper.

        Args:
            env (Env): The environment to wrap
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, min_action=min_action, max_action=max_action
        )

        assert isinstance(env.action_space, Box)
        assert not np.any(env.action_space.low == np.inf) and not np.any(
            env.action_space.high == np.inf
        )

        if not isinstance(min_action, np.ndarray):
            assert np.issubdtype(type(min_action), np.integer) or np.issubdtype(
                type(min_action), np.floating
            )
            min_action = np.full(env.action_space.shape, min_action)

        assert min_action.shape == env.action_space.shape
        assert not np.any(min_action == np.inf)

        if not isinstance(max_action, np.ndarray):
            assert np.issubdtype(type(max_action), np.integer) or np.issubdtype(
                type(max_action), np.floating
            )
            max_action = np.full(env.action_space.shape, max_action)
        assert max_action.shape == env.action_space.shape
        assert not np.any(max_action == np.inf)

        assert isinstance(env.action_space, Box)
        assert np.all(np.less_equal(min_action, max_action))

        # Imagine the x-axis between the old Box and the y-axis being the new Box
        gradient = (env.action_space.high - env.action_space.low) / (
            max_action - min_action
        )
        intercept = gradient * -min_action + env.action_space.low

        LambdaActionV0.__init__(
            self,
            env=env,
            func=lambda action: gradient * action + intercept,
            action_space=Box(
                low=min_action,
                high=max_action,
                shape=env.action_space.shape,
                dtype=env.action_space.dtype,
            ),
        )
