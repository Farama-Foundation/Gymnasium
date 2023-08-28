"""Vectorizes observation wrappers to works for `VectorEnv`."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Sequence

import numpy as np

from gymnasium import Space
from gymnasium.core import Env, ObsType
from gymnasium.vector import VectorEnv, VectorObservationWrapper
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate
from gymnasium.wrappers import transform_observation


class TransformObservation(VectorObservationWrapper):
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
        """Constructor for the transform observation wrapper.

        Args:
            env: The vector environment to wrap
            func: A function that will transform the vector observation. If this transformed observation is outside the observation space of ``env.observation_space`` then provide an ``observation_space``.
            observation_space: The observation spaces of the wrapper, if None, then it is assumed the same as ``env.observation_space``.
        """
        super().__init__(env)

        if observation_space is not None:
            self.observation_space = observation_space

        self.func = func

    def observations(self, observations: ObsType) -> ObsType:
        """Apply function to the vector observation."""
        return self.func(observations)


class VectorizeTransformObservation(VectorObservationWrapper):
    """Vectorizes a single-agent transform observation wrapper for vector environments."""

    class VectorizedEnv(Env):
        """Fake single-agent environment uses for the single-agent wrapper."""

        def __init__(self, observation_space: Space):
            """Constructor for the fake environment."""
            self.observation_space = observation_space

    def __init__(
        self,
        env: VectorEnv,
        wrapper: type[transform_observation.TransformObservation],
        **kwargs: Any,
    ):
        """Constructor for the vectorized transform observation wrapper.

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

    def observations(self, observations: ObsType) -> ObsType:
        """Iterates over the vector observations applying the single-agent wrapper ``observation`` then concatenates the observations together again."""
        if self.same_out:
            return concatenate(
                self.single_observation_space,
                tuple(
                    self.wrapper.func(obs)
                    for obs in iterate(self.observation_space, observations)
                ),
                observations,
            )
        else:
            return deepcopy(
                concatenate(
                    self.single_observation_space,
                    tuple(
                        self.wrapper.func(obs)
                        for obs in iterate(self.env.observation_space, observations)
                    ),
                    self.out,
                )
            )


class FilterObservation(VectorizeTransformObservation):
    """Vector wrapper for filtering dict or tuple observation spaces."""

    def __init__(self, env: VectorEnv, filter_keys: Sequence[str | int]):
        """Constructor for the filter observation wrapper.

        Args:
            env: The vector environment to wrap
            filter_keys: The subspaces to be included, use a list of strings or integers for ``Dict`` and ``Tuple`` spaces respectivesly
        """
        super().__init__(
            env, transform_observation.FilterObservation, filter_keys=filter_keys
        )


class FlattenObservation(VectorizeTransformObservation):
    """Observation wrapper that flattens the observation."""

    def __init__(self, env: VectorEnv):
        """Constructor for any environment's observation space that implements ``spaces.utils.flatten_space`` and ``spaces.utils.flatten``.

        Args:
            env:  The vector environment to wrap
        """
        super().__init__(env, transform_observation.FlattenObservation)


class GrayscaleObservation(VectorizeTransformObservation):
    """Observation wrapper that converts an RGB image to grayscale."""

    def __init__(self, env: VectorEnv, keep_dim: bool = False):
        """Constructor for an RGB image based environments to make the image grayscale.

        Args:
            env: The vector environment to wrap
            keep_dim: If to keep the channel in the observation, if ``True``, ``obs.shape == 3`` else ``obs.shape == 2``
        """
        super().__init__(
            env, transform_observation.GrayscaleObservation, keep_dim=keep_dim
        )


class ResizeObservation(VectorizeTransformObservation):
    """Resizes image observations using OpenCV to shape."""

    def __init__(self, env: VectorEnv, shape: tuple[int, ...]):
        """Constructor that requires an image environment observation space with a shape.

        Args:
            env: The vector environment to wrap
            shape: The resized observation shape
        """
        super().__init__(env, transform_observation.ResizeObservation, shape=shape)


class ReshapeObservation(VectorizeTransformObservation):
    """Reshapes array based observations to shapes."""

    def __init__(self, env: VectorEnv, shape: int | tuple[int, ...]):
        """Constructor for env with Box observation space that has a shape product equal to the new shape product.

        Args:
            env: The vector environment to wrap
            shape: The reshaped observation space
        """
        super().__init__(env, transform_observation.ReshapeObservation, shape=shape)


class RescaleObservation(VectorizeTransformObservation):
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
            transform_observation.RescaleObservation,
            min_obs=min_obs,
            max_obs=max_obs,
        )


class DtypeObservation(VectorizeTransformObservation):
    """Observation wrapper for transforming the dtype of an observation."""

    def __init__(self, env: VectorEnv, dtype: Any):
        """Constructor for Dtype observation wrapper.

        Args:
            env: The vector environment to wrap
            dtype: The new dtype of the observation
        """
        super().__init__(env, transform_observation.DtypeObservation, dtype=dtype)
