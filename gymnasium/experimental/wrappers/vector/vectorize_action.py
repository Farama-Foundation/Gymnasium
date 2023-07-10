"""Vectorizes action wrappers to work for `VectorEnv`."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np

from gymnasium import Space
from gymnasium.core import ActType, Env
from gymnasium.experimental.vector import VectorActionWrapper, VectorEnv
from gymnasium.experimental.wrappers import lambda_action
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate


class LambdaActionV0(VectorActionWrapper):
    """Transforms an action via a function provided to the wrapper.

    The function :attr:`func` will be applied to all vector actions.
    If the observations from :attr:`func` are outside the bounds of the ``env``'s action space, provide an :attr:`action_space`.
    """

    def __init__(
        self,
        env: VectorEnv,
        func: Callable[[ActType], Any],
        action_space: Space | None = None,
    ):
        """Constructor for the lambda action wrapper.

        Args:
            env: The vector environment to wrap
            func: A function that will transform an action. If this transformed action is outside the action space of ``env.action_space`` then provide an ``action_space``.
            action_space: The action spaces of the wrapper, if None, then it is assumed the same as ``env.action_space``.
        """
        super().__init__(env)

        if action_space is not None:
            self.action_space = action_space

        self.func = func

    def actions(self, actions: ActType) -> ActType:
        """Applies the :attr:`func` to the actions."""
        return self.func(actions)


class VectorizeLambdaActionV0(VectorActionWrapper):
    """Vectorizes a single-agent lambda action wrapper for vector environments."""

    class VectorizedEnv(Env):
        """Fake single-agent environment uses for the single-agent wrapper."""

        def __init__(self, action_space: Space):
            """Constructor for the fake environment."""
            self.action_space = action_space

    def __init__(
        self, env: VectorEnv, wrapper: type[lambda_action.LambdaActionV0], **kwargs: Any
    ):
        """Constructor for the vectorized lambda action wrapper.

        Args:
            env: The vector environment to wrap
            wrapper: The wrapper to vectorize
            **kwargs: Arguments for the LambdaActionV0 wrapper
        """
        super().__init__(env)

        self.wrapper = wrapper(
            self.VectorizedEnv(self.env.single_action_space), **kwargs
        )
        self.single_action_space = self.wrapper.action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.same_out = self.action_space == self.env.action_space
        self.out = create_empty_array(self.single_action_space, self.num_envs)

    def actions(self, actions: ActType) -> ActType:
        """Applies the wrapper to each of the action.

        Args:
            actions: The actions to apply the function to

        Returns:
            The updated actions using the wrapper func
        """
        if self.same_out:
            return concatenate(
                self.single_action_space,
                tuple(
                    self.wrapper.func(action)
                    for action in iterate(self.action_space, actions)
                ),
                actions,
            )
        else:
            return deepcopy(
                concatenate(
                    self.single_action_space,
                    tuple(
                        self.wrapper.func(action)
                        for action in iterate(self.action_space, actions)
                    ),
                    self.out,
                )
            )


class ClipActionV0(VectorizeLambdaActionV0):
    """Clip the continuous action within the valid :class:`Box` observation space bound."""

    def __init__(self, env: VectorEnv):
        """Constructor for the Clip Action wrapper.

        Args:
            env: The vector environment to wrap
        """
        super().__init__(env, lambda_action.ClipActionV0)


class RescaleActionV0(VectorizeLambdaActionV0):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action]."""

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
            lambda_action.RescaleActionV0,
            min_action=min_action,
            max_action=max_action,
        )
