"""Vectorizes action wrappers to work for `VectorEnv`."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np

from gymnasium import Space
from gymnasium.core import ActType, Env
from gymnasium.logger import warn
from gymnasium.vector import VectorActionWrapper, VectorEnv
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate
from gymnasium.wrappers import transform_action


class TransformAction(VectorActionWrapper):
    """Transforms an action via a function provided to the wrapper.

    The function :attr:`func` will be applied to all vector actions.
    If the observations from :attr:`func` are outside the bounds of the ``env``'s action space,
    provide an :attr:`action_space` which specifies the action space for the vectorized environment.

    Example - Without action transformation:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> for _ in range(10):
        ...     obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
        ...
        >>> envs.close()
        >>> obs
        array([[-0.46553135, -0.00142543],
               [-0.498371  , -0.00715587],
               [-0.46515748, -0.00624371]], dtype=float32)

    Example - With action transformation:
        >>> import gymnasium as gym
        >>> from gymnasium.spaces import Box
        >>> def shrink_action(act):
        ...     return act * 0.3
        ...
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> new_action_space = Box(low=shrink_action(envs.action_space.low), high=shrink_action(envs.action_space.high))
        >>> envs = TransformAction(env=envs, func=shrink_action, action_space=new_action_space)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> for _ in range(10):
        ...     obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
        ...
        >>> envs.close()
        >>> obs
        array([[-0.48468155, -0.00372536],
               [-0.47599354, -0.00545912],
               [-0.46543318, -0.00615723]], dtype=float32)
    """

    def __init__(
        self,
        env: VectorEnv,
        func: Callable[[ActType], Any],
        action_space: Space | None = None,
        single_action_space: Space | None = None,
    ):
        """Constructor for the lambda action wrapper.

        Args:
            env: The vector environment to wrap
            func: A function that will transform an action. If this transformed action is outside the action space of ``env.action_space`` then provide an ``action_space``.
            action_space: The action spaces of the wrapper. If None, then it is computed from ``single_action_space``. If ``single_action_space`` is not provided either, then it is assumed to be the same as ``env.action_space``.
            single_action_space: The action space of the non-vectorized environment. If None, then it is assumed the same as ``env.single_action_space``.
        """
        super().__init__(env)

        if action_space is None:
            if single_action_space is not None:
                self.single_action_space = single_action_space
                self.action_space = batch_space(single_action_space, self.num_envs)
        else:
            self.action_space = action_space
            if single_action_space is not None:
                self.single_action_space = single_action_space
            # TODO: We could compute single_action_space from the action_space if only the latter is provided and avoid the warning below.
        if self.action_space != batch_space(self.single_action_space, self.num_envs):
            warn(
                f"For {env}, the action space and the batched single action space don't match as expected, action_space={env.action_space}, batched single_action_space={batch_space(self.single_action_space, self.num_envs)}"
            )

        self.func = func

    def actions(self, actions: ActType) -> ActType:
        """Applies the :attr:`func` to the actions."""
        return self.func(actions)


class VectorizeTransformAction(VectorActionWrapper):
    """Vectorizes a single-agent transform action wrapper for vector environments.

    Example - Without action transformation:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
        >>> envs.close()
        >>> obs
        array([[-4.6343064e-01,  9.8971417e-05],
               [-4.4488689e-01, -1.9375233e-03],
               [-4.3118435e-01, -1.5342437e-03]], dtype=float32)

    Example - Adding a transform that applies a ReLU to the action:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformAction
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> envs = VectorizeTransformAction(envs, wrapper=TransformAction, func=lambda x: (x > 0.0) * x, action_space=envs.single_action_space)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
        >>> envs.close()
        >>> obs
        array([[-4.6343064e-01,  9.8971417e-05],
               [-4.4354835e-01, -5.9898634e-04],
               [-4.3034542e-01, -6.9532328e-04]], dtype=float32)
    """

    class _SingleEnv(Env):
        """Fake single-agent environment used for the single-agent wrapper."""

        def __init__(self, action_space: Space):
            """Constructor for the fake environment."""
            self.action_space = action_space

    def __init__(
        self,
        env: VectorEnv,
        wrapper: type[transform_action.TransformAction],
        **kwargs: Any,
    ):
        """Constructor for the vectorized lambda action wrapper.

        Args:
            env: The vector environment to wrap
            wrapper: The wrapper to vectorize
            **kwargs: Arguments for the LambdaAction wrapper
        """
        super().__init__(env)

        self.wrapper = wrapper(self._SingleEnv(self.env.single_action_space), **kwargs)
        self.single_action_space = self.wrapper.action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.same_out = self.action_space == self.env.action_space
        self.out = create_empty_array(self.env.single_action_space, self.num_envs)

    def actions(self, actions: ActType) -> ActType:
        """Applies the wrapper to each of the action.

        Args:
            actions: The actions to apply the function to

        Returns:
            The updated actions using the wrapper func
        """
        if self.same_out:
            return concatenate(
                self.env.single_action_space,
                tuple(
                    self.wrapper.func(action)
                    for action in iterate(self.action_space, actions)
                ),
                actions,
            )
        else:
            return deepcopy(
                concatenate(
                    self.env.single_action_space,
                    tuple(
                        self.wrapper.func(action)
                        for action in iterate(self.action_space, actions)
                    ),
                    self.out,
                )
            )


class ClipAction(VectorizeTransformAction):
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example - Passing an out-of-bounds action to the environment to be clipped.
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> envs = ClipAction(envs)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> obs, rew, term, trunc, info = envs.step(np.array([5.0, -5.0, 2.0]))
        >>> envs.close()
        >>> obs
        array([[-0.4624777 ,  0.00105192],
               [-0.44504836, -0.00209899],
               [-0.42884544,  0.00080468]], dtype=float32)
    """

    def __init__(self, env: VectorEnv):
        """Constructor for the Clip Action wrapper.

        Args:
            env: The vector environment to wrap
        """
        super().__init__(env, transform_action.ClipAction)


class RescaleAction(VectorizeTransformAction):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    Example - Without action scaling:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> for _ in range(10):
        ...     obs, rew, term, trunc, info = envs.step(0.5 * np.ones((3, 1)))
        ...
        >>> envs.close()
        >>> obs
        array([[-0.44799727,  0.00266526],
               [-0.4351738 ,  0.00133522],
               [-0.42683297,  0.00048403]], dtype=float32)

    Example - With action scaling:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
        >>> envs = RescaleAction(envs, 0.0, 1.0)
        >>> _ = envs.action_space.seed(123)
        >>> obs, info = envs.reset(seed=123)
        >>> for _ in range(10):
        ...     obs, rew, term, trunc, info = envs.step(0.5 * np.ones((3, 1)))
        ...
        >>> envs.close()
        >>> obs
        array([[-0.48657528, -0.00395268],
               [-0.47377947, -0.00529102],
               [-0.46546045, -0.00614867]], dtype=float32)
    """

    def __init__(
        self,
        env: VectorEnv,
        min_action: float | int | np.ndarray,
        max_action: float | int | np.ndarray,
    ):
        """Initializes the :class:`RescaleAction` wrapper.

        Args:
            env (Env): The vector environment to wrap
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        super().__init__(
            env,
            transform_action.RescaleAction,
            min_action=min_action,
            max_action=max_action,
        )
