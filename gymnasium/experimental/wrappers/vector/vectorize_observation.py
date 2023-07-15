"""Vectorizes observation wrappers to works for `VectorEnv`."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Sequence

import numpy as np

from gymnasium import Space
from gymnasium.core import Env, ObsType
from gymnasium.experimental.vector import VectorEnv, VectorObservationWrapper
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
        vector_func: Callable[[ObsType], Any],
        single_func: Callable[[ObsType], Any],
        observation_space: Space | None = None,
    ):
        """Constructor for the lambda observation wrapper.

        Args:
            env: The vector environment to wrap
            vector_func: A function that will transform the vector observation. If this transformed observation is outside the observation space of ``env.observation_space`` then provide an ``observation_space``.
            single_func: A function that will transform an individual observation.
            observation_space: The observation spaces of the wrapper, if None, then it is assumed the same as ``env.observation_space``.
        """
        super().__init__(env)

        if observation_space is not None:
            self.observation_space = observation_space

        self.vector_func = vector_func
        self.single_func = single_func

    def vector_observation(self, observation: ObsType) -> ObsType:
        """Apply function to the vector observation."""
        return self.vector_func(observation)

    def single_observation(self, observation: ObsType) -> ObsType:
        """Apply function to the single observation."""
        return self.single_func(observation)


class VectorizeLambdaObservationV0(VectorObservationWrapper):
    """Vectori`es a single-agent lambda observation wrapper for vector environments."""

    class VectorizedEnv(Env):
        """Fake single-agent environment uses for the single-agent wrapper."""

        def __init__(self, observation_space: Space):
            """Constructor for the fake environment."""
            self.observation_space = observation_space

    def __init__(
        self,
        env: VectorEnv,
        wrapper: type[lambda_observation.LambdaObservationV0],
        **kwargs: Any,
    ):
        """Constructor for the vectorized lambda observation wrapper.

        Args:
            env: The vector environment to wrap.
            wrapper: The wrapper to vectorize
            **kwargs: Keyword argument for the wrapper
        """
        super().__init__(env)

        self.wrapper = wrapper(
            self.VectorizedEnv(self.env.single_observation_space), **kwargs
        )
        self.single_observation_space = self.wrapper.observation_space
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

        self.same_out = self.observation_space == self.env.observation_space
        self.out = create_empty_array(self.single_observation_space, self.num_envs)

    def vector_observation(self, observation: ObsType) -> ObsType:
        """Iterates over the vector observations applying the single-agent wrapper ``observation`` then concatenates the observations together again."""
        if self.same_out:
            return concatenate(
                self.single_observation_space,
                tuple(
                    self.wrapper.func(obs)
                    for obs in iterate(self.observation_space, observation)
                ),
                observation,
            )
        else:
            return deepcopy(
                concatenate(
                    self.single_observation_space,
                    tuple(
                        self.wrapper.func(obs)
                        for obs in iterate(self.observation_space, observation)
                    ),
                    self.out,
                )
            )

    def single_observation(self, observation: ObsType) -> ObsType:
        """Transforms a single observation using the wrapper transformation function."""
        return self.wrapper.func(observation)


class FilterObservationV0(VectorizeLambdaObservationV0):
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


class FlattenObservationV0(VectorizeLambdaObservationV0):
    """Observation wrapper that flattens the observation."""

    def __init__(self, env: VectorEnv):
        """Constructor for any environment's observation space that implements ``spaces.utils.flatten_space`` and ``spaces.utils.flatten``.

        Args:
            env:  The vector environment to wrap
        """
        super().__init__(env, lambda_observation.FlattenObservationV0)


class GrayscaleObservationV0(VectorizeLambdaObservationV0):
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


class ResizeObservationV0(VectorizeLambdaObservationV0):
    """Resizes image observations using OpenCV to shape."""

    def __init__(self, env: VectorEnv, shape: tuple[int, ...]):
        """Constructor that requires an image environment observation space with a shape.

        Args:
            env: The vector environment to wrap
            shape: The resized observation shape
        """
        super().__init__(env, lambda_observation.ResizeObservationV0, shape=shape)


class ReshapeObservationV0(VectorizeLambdaObservationV0):
    """Reshapes array based observations to shapes."""

    def __init__(self, env: VectorEnv, shape: int | tuple[int, ...]):
        """Constructor for env with Box observation space that has a shape product equal to the new shape product.

        Args:
            env: The vector environment to wrap
            shape: The reshaped observation space
        """
        super().__init__(env, lambda_observation.ReshapeObservationV0, shape=shape)


class RescaleObservationV0(VectorizeLambdaObservationV0):
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


class DtypeObservationV0(VectorizeLambdaObservationV0):
    """Observation wrapper for transforming the dtype of an observation."""

    def __init__(self, env: VectorEnv, dtype: Any):
        """Constructor for Dtype observation wrapper.

        Args:
            env: The vector environment to wrap
            dtype: The new dtype of the observation
        """
        super().__init__(env, lambda_observation.DtypeObservationV0, dtype=dtype)
