"""A collection of vector based lambda action wrappers.

* ``LambdaActionV0`` - Transforms the actions based on a function
* ``VectoriseLambdaObservationV0`` - Vectorises a single agent lambda action wrapper
* ``ClipActionV0`` - Clips the action within a bounds
* ``RescaleActionV0`` - Rescales the action within a minimum and maximum actions
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np

from gymnasium import Space
from gymnasium.core import ActType, Env
from gymnasium.experimental import VectorEnv, wrappers
from gymnasium.experimental.vector import VectorActionWrapper
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


class VectoriseLambdaActionV0(VectorActionWrapper):
    """Vectorises a single-agent lambda action wrapper for vector environments."""

    class VectorisedEnv(Env):
        """Fake single-agent environment uses for the single-agent wrapper."""

        def __init__(self, action_space: Space):
            """Constructor for the fake environment."""
            self.action_space = action_space

    def __init__(
        self, env: VectorEnv, wrapper: type[wrappers.LambdaActionV0], **kwargs: Any
    ):
        """Constructor for the vectorised lambda action wrapper.

        Args:
            env: The vector environment to wrap
            wrapper: The wrapper to vectorise
            **kwargs: Arguments for the LambdaActionV0 wrapper
        """
        super().__init__(env)

        self.wrapper = wrapper(
            self.VectorisedEnv(self.env.single_action_space), **kwargs
        )
        self.single_action_space = self.wrapper.action
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.out = create_empty_array(self.single_action_space, self.num_envs)

    def actions(self, actions: ActType) -> ActType:
        """Applies the wrapper to each of the action.

        Args:
            actions: The actions to apply the function to

        Returns:
            The updated actions using the wrapper func
        """
        return deepcopy(
            concatenate(
                self.single_action_space,
                (
                    self.wrapper.func(action)
                    for action in iterate(self.action_space, actions)
                ),
                self.out,
            )
        )


class ClipActionV0(VectoriseLambdaActionV0):
    """Clip the continuous action within the valid :class:`Box` observation space bound."""

    def __init__(self, env: VectorEnv):
        """Constructor for the Clip Action wrapper.

        Args:
            env: The vector environment to wrap
        """
        super().__init__(env, wrappers.ClipActionV0)


class RescaleActionV0(VectoriseLambdaActionV0):
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
            env, wrappers.RescaleActionV0, min_action=min_action, max_action=max_action
        )
