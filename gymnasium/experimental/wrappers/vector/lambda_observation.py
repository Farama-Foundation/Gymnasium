"""A collection of vector based lambda observation wrappers.

* ``LambdaObservationV0`` - Transforms the observation with a function
* ``VectoriseLambdaObservationV0`` - Vectorises a single agent lambda observation wrapper
* ``FilterObservationV0`` - Filters a ``Tuple`` or ``Dict`` to only include certain keys
* ``FlattenObservationV0`` - Flattens the observations
* ``GrayscaleObservationV0`` - Converts a RGB observation to a grayscale observation
* ``ResizeObservationV0`` - Resizes an array-based observation (normally a RGB observation)
* ``ReshapeObservationV0`` - Reshapes an array-based observation
* ``RescaleObservationV0`` - Rescales an observation to between a minimum and maximum value
* ``DtypeObservationV0`` - Convert an observation to a dtype
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Sequence

import numpy as np

from gymnasium import Space
from gymnasium.core import Env, ObsType
from gymnasium.experimental import VectorEnv, wrappers
from gymnasium.experimental.vector import VectorObservationWrapper
from gymnasium.experimental.vector.utils import batch_space, concatenate, iterate
from gymnasium.experimental.wrappers import lambda_observation
from gymnasium.vector.utils import create_empty_array


class LambdaObservationV0(VectorObservationWrapper):
    """Transforms an observation via a function provided to the wrapper.

    The function :attr:`func` will be applied to all vector observations.
    If the observations from :attr:`func` are outside the bounds of the ``env``'s observation space, provide an :attr:`observation_space`.
    """

    def __init__(
        self,
        env: VectorEnv,
        func: Callable[[ObsType], Any],
        observation_space: Space | None = None,
    ):
        """Constructor for the lambda observation wrapper.

        Args:
            env: The vector environment to wrap
            func: A function that will transform an observation. If this transformed observation is outside the observation space of `env.observation_space` then provide an `observation_space`.
            observation_space: The observation spaces of the wrapper, if None, then it is assumed the same as `env.observation_space`.
        """
        super().__init__(env)

        if observation_space is not None:
            self.observation_space = observation_space

        self.func = func

    def observation(self, observation: ObsType) -> ObsType:
        """Apply function to the observation."""
        return self.func(observation)


class VectoriseLambdaObservationV0(VectorObservationWrapper):
    """Vectorises a single-agent lambda action wrapper for vector environments."""

    class VectorisedEnv(Env):
        """Fake single-agent environment uses for the single-agent wrapper."""

        def __init__(self, observation_space: Space):
            """Constructor for the fake environment."""
            self.observation_space = observation_space

    def __init__(
        self, env: VectorEnv, wrapper: type[wrappers.LambdaObservationV0], **kwargs: Any
    ):
        """Constructor for the vectorised lambda observation wrapper.

        Args:
            env: The vector environment to wrap.
            wrapper: The wrapper to vectorise
            **kwargs: Keyword argument for the wrapper
        """
        super().__init__(env)

        self.wrapper = wrapper(
            self.VectorisedEnv(self.env.single_observation_space), **kwargs
        )
        self.single_observation_space = self.wrapper.observation_space
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.out = create_empty_array(self.single_observation_space, self.num_envs)

    def observation(self, observation: ObsType) -> ObsType:
        """Iterates over the vector observations applying the single-agent wrapper ``observation`` then concatenates the observations together again."""
        return deepcopy(
            concatenate(
                self.single_observation_space,
                (
                    self.wrapper.observation(obs)
                    for obs in iterate(self.observation_space, observation)
                ),
                self.out,
            )
        )


class FilterObservationV0(VectoriseLambdaObservationV0):
    """Vector wrapper for filtering dict or tuple observation spaces."""

    def __init__(self, env: VectorEnv, filter_keys: Sequence[str | int]):
        """Constructor for the filter observation wrapper.

        Args:
            env: The vector environment to wrap
            filter_keys: The subspaces to be included, use a list of strings or integers for ``Dict`` and ``Tuple`` spaces respectivesly
        """
        super().__init__(
            env, lambda_observation.FilterObservationV0, filter_keys=filter_keys
        )


class FlattenObservationV0(VectoriseLambdaObservationV0):
    """Observation wrapper that flattens the observation."""

    def __init__(self, env: VectorEnv):
        """Constructor for any environment's observation space that implements ``spaces.utils.flatten_space`` and ``spaces.utils.flatten``.

        Args:
            env:  The vector environment to wrap
        """
        super().__init__(env, lambda_observation.FlattenObservationV0)


class GrayscaleObservationV0(VectoriseLambdaObservationV0):
    """Observation wrapper that converts an RGB image to grayscale."""

    def __init__(self, env: VectorEnv, keep_dim: bool = False):
        """Constructor for an RGB image based environments to make the image grayscale.

        Args:
            env: The vector environment to wrap
            keep_dim: If to keep the channel in the observation, if ``True``, ``obs.shape == 3`` else ``obs.shape == 2``
        """
        super().__init__(
            env, lambda_observation.GrayscaleObservationV0, keep_dim=keep_dim
        )


class ResizeObservationV0(VectoriseLambdaObservationV0):
    """Resizes image observations using OpenCV to shape."""

    def __init__(self, env: VectorEnv, shape: tuple[int, ...]):
        """Constructor that requires an image environment observation space with a shape.

        Args:
            env: The vector environment to wrap
            shape: The resized observation shape
        """
        super().__init__(env, lambda_observation.ResizeObservationV0, shape=shape)


class ReshapeObservationV0(VectoriseLambdaObservationV0):
    """Reshapes array based observations to shapes."""

    def __init__(self, env: VectorEnv, shape: int | tuple[int, ...]):
        """Constructor for env with Box observation space that has a shape product equal to the new shape product.

        Args:
            env: The vector environment to wrap
            shape: The reshaped observation space
        """
        super().__init__(env, lambda_observation.ReshapeObservationV0, shape=shape)


class RescaleObservationV0(VectoriseLambdaObservationV0):
    """Linearly rescales observation to between a minimum and maximum value."""

    def __init__(
        self,
        env: VectorEnv,
        min_obs: np.floating | np.integer | np.ndarray,
        max_obs: np.floating | np.integer | np.ndarray,
    ):
        """Constructor that requires the env observation spaces to be a :class:`Box`.

        Args:
            env: The vector environment to wrap
            min_obs: The new minimum observation bound
            max_obs: The new maximum observation bound
        """
        super().__init__(
            env,
            lambda_observation.RescaleObservationV0,
            min_obs=min_obs,
            max_obs=max_obs,
        )


class DtypeObservationV0(VectoriseLambdaObservationV0):
    """Observation wrapper for transforming the dtype of an observation."""

    def __init__(self, env: VectorEnv, dtype: Any):
        """Constructor for Dtype observation wrapper.

        Args:
            env: The vector environment to wrap
            dtype: The new dtype of the observation
        """
        super().__init__(env, lambda_observation.DtypeObservationV0, dtype=dtype)


# class PixelObservationV0(VectoriseLambdaObservationV0):
#     def __init__(
#         self,
#         env: VectorEnv,
#         pixels_only: bool = True,
#         pixels_key: str = "pixels",
#         obs_key: str = "state",
#     ):
#         super().__init__(
#             env,
#             lambda_observation.PixelObservationV0,
#             pixels_only=pixels_only,
#             pixels_key=pixels_key,
#             obs_key=obs_key,
#         )
