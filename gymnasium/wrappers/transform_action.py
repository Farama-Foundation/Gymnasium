"""A collection of wrappers that all use the LambdaAction class.

* ``TransformAction`` - Transforms the actions based on a function
* ``ClipAction`` - Clips the action within a bounds
* ``DiscretizeAction`` - Discretizes a continuous Box action space into a single Discrete space
* ``RescaleAction`` - Rescales the action within a minimum and maximum actions
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperActType
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Space


__all__ = ["TransformAction", "ClipAction", "RescaleAction"]

from gymnasium.wrappers.utils import rescale_box


class TransformAction(
    gym.ActionWrapper[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Applies a function to the ``action`` before passing the modified value to the environment ``step`` function.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.TransformAction`.

    Example:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> env = gym.make("MountainCarContinuous-v0")
        >>> _ = env.reset(seed=123)
        >>> obs, *_= env.step(np.array([0.0, 1.0]))
        >>> obs
        array([-4.6397772e-01, -4.4808415e-04], dtype=float32)
        >>> env = gym.make("MountainCarContinuous-v0")
        >>> env = TransformAction(env, lambda a: 0.5 * a + 0.1, env.action_space)
        >>> _ = env.reset(seed=123)
        >>> obs, *_= env.step(np.array([0.0, 1.0]))
        >>> obs
        array([-4.6382770e-01, -2.9808417e-04], dtype=float32)

    Change logs:
     * v1.0.0 - Initially added
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[WrapperActType], ActType],
        action_space: Space[WrapperActType] | None,
    ):
        """Initialize TransformAction.

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


class ClipAction(
    TransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Clips the ``action`` pass to ``step`` to be within the environment's `action_space`.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.ClipAction`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ClipAction
        >>> import numpy as np
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-inf, inf, (3,), float32)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([5.0, -2.0, 0.0], dtype=np.float32))
        ... # Executes the action np.array([1.0, -1.0, 0]) in the base environment

    Change logs:
     * v0.12.6 - Initially added
     * v1.0.0 - Action space is updated to infinite bounds as is technically correct
    """

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to wrap
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        TransformAction.__init__(
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


class RescaleAction(
    TransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Affinely (linearly) rescales a ``Box`` action space of the environment to within the range of ``[min_action, max_action]``.

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.RescaleAction`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import RescaleAction
        >>> import numpy as np
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> _ = env.reset(seed=42)
        >>> obs, _, _, _, _ = env.step(np.array([1, 1, 1], dtype=np.float32))
        >>> _ = env.reset(seed=42)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 0.75], dtype=np.float32)
        >>> wrapped_env = RescaleAction(env, min_action=min_action, max_action=max_action)
        >>> wrapped_env_obs, _, _, _, _ = wrapped_env.step(max_action)
        >>> np.all(obs == wrapped_env_obs)
        np.True_

    Change logs:
     * v0.15.4 - Initially added
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        min_action: np.floating | np.integer | np.ndarray,
        max_action: np.floating | np.integer | np.ndarray,
    ):
        """Constructor for the Rescale Action wrapper.

        Args:
            env (Env): The environment to wrap
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(
            self, min_action=min_action, max_action=max_action
        )

        act_space, _, func = rescale_box(env.action_space, min_action, max_action)
        TransformAction.__init__(
            self,
            env=env,
            func=func,
            action_space=act_space,
        )


class DiscretizeAction(
    TransformAction[ObsType, WrapperActType, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Uniformly discretizes a continuous Box action space into a single Discrete space.

    Example 1 - Discretize Pendulum action space:
        >>> env = gym.make("Pendulum-v1")
        >>> env.action_space
        Box(-2.0, 2.0, (1,), float32)
        >>> obs, _ = env.reset(seed=42)
        >>> obs, *_ = env.step([-0.6])
        >>> obs
        array([-0.17606162,  0.9843792 ,  0.5292768 ], dtype=float32)
        >>> env = DiscretizeAction(env, bins=10)
        >>> env.action_space
        Discrete(10)
        >>> obs, _ = env.reset(seed=42)
        >>> obs, *_ = env.step(3)
        >>> obs
        array([-0.17606162,  0.9843792 ,  0.5292768 ], dtype=float32)

    Example 2 - Discretize Reacher action space:
        >>> env = gym.make("Reacher-v5")
        >>> env.action_space
        Box(-1.0, 1.0, (2,), float32)
        >>> obs, _ = env.reset(seed=42)
        >>> obs, *_ = env.step([-0.3, -0.5])
        >>> obs
        array([ 0.99908342,  0.99948506,  0.04280567, -0.03208766,  0.10445588,
                0.11442572, -1.18958125, -1.97979484,  0.1054461 , -0.10896341])
        >>> env = DiscretizeAction(env, bins=10)
        >>> env.action_space
        Discrete(100)
        >>> obs, _ = env.reset(seed=42)
        >>> obs, *_ = env.step(32)
        >>> obs
        array([ 0.99908342,  0.99948506,  0.04280567, -0.03208766,  0.10445588,
                0.11442572, -1.18958118, -1.97979484,  0.1054461 , -0.10896341])

    Example 2 - Discretize Reacher action space with MultiDiscrete:
        >>> env = gym.make("Reacher-v5")
        >>> env.action_space
        Box(-1.0, 1.0, (2,), float32)
        >>> obs, _ = env.reset(seed=42)
        >>> obs, *_ = env.step([-0.3, -0.5])
        >>> obs
        array([ 0.99908342,  0.99948506,  0.04280567, -0.03208766,  0.10445588,
                0.11442572, -1.18958125, -1.97979484,  0.1054461 , -0.10896341])
        >>> env = DiscretizeAction(env, bins=10, multidiscrete=True)
        >>> env.action_space
        MultiDiscrete([10 10])
        >>> obs, _ = env.reset(seed=42)
        >>> obs, *_ = env.step([3, 2])
        >>> obs
        array([ 0.99908342,  0.99948506,  0.04280567, -0.03208766,  0.10445588,
                0.11442572, -1.18958118, -1.97979484,  0.1054461 , -0.10896341])
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        bins: int | tuple[int, ...],
        multidiscrete: bool = False,
    ):
        """Constructor for the discretize action wrapper.

        Args:
            env: The environment to wrap.
            bins: int or tuple of ints (number of bins per dimension).
            multidiscrete: If True, use MultiDiscrete action space instead of flattening to Discrete.
        """
        if not isinstance(env.action_space, Box):
            raise TypeError(
                "DiscretizeAction is only compatible with Box continuous actions."
            )

        self.low = env.action_space.low
        self.high = env.action_space.high
        self.n_dims = self.low.shape[0]

        if np.any(np.isinf(self.low)) or np.any(np.isinf(self.high)):
            raise ValueError(
                "Discretization requires action space to be finite. "
                f"Found: low={self.low}, high={self.high}"
            )

        self.multidiscrete = multidiscrete
        gym.utils.RecordConstructorArgs.__init__(self, bins=bins)
        gym.ActionWrapper.__init__(self, env)

        if isinstance(bins, int):
            self.bins = np.array([bins] * self.n_dims)
        else:
            assert (
                len(bins) == self.n_dims
            ), f"bins must match action dimensions: expected {self.n_dims}, got {len(bins)}"
            self.bins = np.array(bins)

        self.bin_centers = [
            0.5
            * (
                np.linspace(self.low[i], self.high[i], self.bins[i] + 1)[:-1]
                + np.linspace(self.low[i], self.high[i], self.bins[i] + 1)[1:]
            )
            for i in range(self.n_dims)
        ]

        if self.multidiscrete:
            self.action_space = MultiDiscrete(self.bins)
        else:
            self.action_space = Discrete(np.prod(self.bins))

    def action(self, act):
        """Discretizes the action."""
        if self.multidiscrete:
            indices = np.asarray(act, dtype=int)
        else:
            indices = self._unflatten_index(act)
        centers = [
            self.bin_centers[i][min(max(idx, 0), self.bins[i] - 1)]
            for i, idx in enumerate(indices)
        ]
        return np.array(centers, dtype=self.env.action_space.dtype)

    def revert_action(self, action):
        """Converts a discretized action to a possible continuous action (the center of the closest bin)."""
        indices = [
            np.argmin(np.abs(self.bin_centers[i] - action[i]))
            for i in range(self.n_dims)
        ]
        if self.multidiscrete:
            return np.array(indices, dtype=int)
        else:
            return np.ravel_multi_index(indices, self.bins)

    def _unflatten_index(self, flat_index):
        indices = []
        for b in reversed(self.bins):
            indices.append(flat_index % b)
            flat_index //= b
        return list(reversed(indices))
