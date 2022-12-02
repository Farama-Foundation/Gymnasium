"""Lambda action wrapper which apply a function to the provided action."""
from typing import Any, Callable, Union

import jumpy as jp
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType
from gymnasium.experimental.wrappers import ArgType


class LambdaActionV0(gym.ActionWrapper):
    """A wrapper that provides a function to modify the action passed to :meth:`step`."""

    def __init__(
        self,
        env: gym.Env,
        func: Callable[[ArgType], Any],
    ):
        """Initialize LambdaAction.

        Args:
            env (Env): The gymnasium environment
            func (Callable): function to apply to action
        """
        super().__init__(env)

        self.func = func

    def action(self, action: ActType) -> Any:
        """Apply function to action."""
        return self.func(action)


class ClipActionV0(LambdaActionV0):
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> env = gym.make('BipedalWalker-v3', disable_env_checker=True)
        >>> env = ClipActionV0(env)
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env.step(np.array([5.0, 2.0, -10.0, 0.0]))
        # Executes the action np.array([1.0, 1.0, -1.0, 0]) in the base environment
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, spaces.Box)
        super().__init__(
            env,
            lambda action: jp.clip(action, env.action_space.low, env.action_space.high),
        )

        self.action_space = spaces.Box(-np.inf, np.inf, env.action_space.shape)


class RescaleActionV0(LambdaActionV0):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> env = gym.make('BipedalWalker-v3', disable_env_checker=True)
        >>> _ = env.reset(seed=42)
        >>> obs, _, _, _, _ = env.step(np.array([1,1,1,1]))
        >>> _ = env.reset(seed=42)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 1.0, 0.75])
        >>> wrapped_env = RescaleActionV0(env, min_action=min_action, max_action=max_action)
        >>> wrapped_env_obs, _, _, _, _ = wrapped_env.step(max_action)
        >>> np.alltrue(obs == wrapped_env_obs)
        True
    """

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, np.ndarray],
        max_action: Union[float, int, np.ndarray],
    ):
        """Initializes the :class:`RescaleAction` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(
            env.action_space, spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        low = env.action_space.low
        high = env.action_space.high

        self.min_action = np.full(
            env.action_space.shape, min_action, dtype=env.action_space.dtype
        )
        self.max_action = np.full(
            env.action_space.shape, max_action, dtype=env.action_space.dtype
        )

        super().__init__(
            env,
            lambda action: jp.clip(
                low
                + (high - low)
                * ((action - self.min_action) / (self.max_action - self.min_action)),
                low,
                high,
            ),
        )
