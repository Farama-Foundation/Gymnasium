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


class ClipActionV0(gym.ActionWrapper):
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
        super().__init__(env)

        self.action_space = spaces.Box(-np.inf, np.inf, env.action_space.shape)

    def action(self, action: ActType) -> jp.ndarray:
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return jp.clip(action, self.action_space.low, self.action_space.high)


class RescaleActionV0(gym.ActionWrapper):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> env = gym.make('BipedalWalker-v3', disable_env_checker=True)
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 1.0, 0.75])
        >>> env = RescaleActionV0(env, min_action=min_action, max_action=max_action)
        >>> env.action_space
        Box(-0.5, [0.   0.5  1.   0.75], (4,), float32)
        >>> RescaleAction(env, min_action, max_action).action_space == gym.spaces.Box(min_action, max_action)
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

        super().__init__(env)
        self.min_action = np.full(
            env.action_space.shape, min_action, dtype=env.action_space.dtype
        )
        self.max_action = np.full(
            env.action_space.shape, max_action, dtype=env.action_space.dtype
        )

        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action: ActType) -> jp.ndarray:
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        return jp.clip(action, low, high)
