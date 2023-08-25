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

    This function allows the manual specification of the vector-observation function as well as the single-observation function.
    This is desirable when, for example, it is possible to process vector observations in parallel or via other more optimized methods.
    Otherwise, the ``VectorizeTransformObservation`` should be used instead, where only ``single_func`` needs to be defined.

    Example:
        Without observation transformation:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs
        array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598],
               [ 0.03517495, -0.000635  , -0.01098382, -0.03203924]],
              dtype=float32)

        With observation transformation:
        >>> import gymnasium as gym
        >>> from gymnasium.spaces import Box
        >>> def scale_and_shift(obs):
        ...     return (obs - 1.0) * 2.0
        ...
        >>> def vector_scale_and_shift(obs):
        ...     return (obs - 1.0) * 2.0
        ...
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3)
        >>> new_obs_space = Box(low=envs.observation_space.low, high=envs.observation_space.high)
        >>> envs = TransformObservation(envs, single_func=scale_and_shift, vector_func=vector_scale_and_shift)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs
        array([[-1.9635296, -2.0892358, -2.055928 , -2.0631256],
               [-1.9429494, -1.9428282, -1.9061728, -1.9503881],
               [-1.9296501, -2.00127  , -2.0219676, -2.0640786]], dtype=float32)
    """

    def __init__(
        self,
        env: VectorEnv,
        vector_func: Callable[[ObsType], Any],
        single_func: Callable[[ObsType], Any],
        observation_space: Space | None = None,
    ):
        """Constructor for the transform observation wrapper.

        Args:
            env: The vector environment to wrap
            vector_func: A function that will transform the vector observation. If this transformed observation is outside the observation space of ``env.observation_space`` then provide an ``observation_space``.
            single_func: A function that will transform an individual observation, this function will be used for the final observation from the environment and is returned under ``info`` and not the normal observation.
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


class VectorizeTransformObservation(VectorObservationWrapper):
    """Vectorizes a single-agent transform observation wrapper for vector environments.

    Most of the lambda observation wrappers for single agent environments have vectorized implementations,
    it is advised that users simply use those instead via importing from `gymnasium.wrappers.vector...`.
    The following example illustrate use-cases where a custom lambda observation wrapper is required.

    Example:
        The normal observation:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs
        array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598],
               [ 0.03517495, -0.000635  , -0.01098382, -0.03203924]],
              dtype=float32)

        Applying a custom lambda observation wrapper that duplicates the observation from the environment
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> from gymnasium.spaces import Box
        >>> from gymnasium.wrappers import TransformObservation
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3)
        >>> old_space = envs.single_observation_space
        >>> new_space = Box(low=np.array([old_space.low, old_space.low]), high=np.array([old_space.high, old_space.high]))
        >>> envs = VectorizeTransformObservation(envs, wrapper=TransformObservation, func=lambda x: np.array([x, x]), observation_space=new_space)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs
        array([[[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
        <BLANKLINE>
               [[ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598],
                [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598]],
        <BLANKLINE>
               [[ 0.03517495, -0.000635  , -0.01098382, -0.03203924],
                [ 0.03517495, -0.000635  , -0.01098382, -0.03203924]]],
              dtype=float32)
    """

    class _SingleEnv(Env):
        """Fake single-agent environment used for the single-agent wrapper."""

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
            self._SingleEnv(self.env.single_observation_space), **kwargs
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
                        for obs in iterate(self.env.observation_space, observation)
                    ),
                    self.out,
                )
            )

    def single_observation(self, observation: ObsType) -> ObsType:
        """Transforms a single observation using the wrapper transformation function."""
        return self.wrapper.func(observation)


class FilterObservation(VectorizeTransformObservation):
    """Vector wrapper for filtering dict or tuple observation spaces.

    Example:
        Create a vectorized environment with a Dict space to demonstrate how to filter keys:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> from gymnasium.spaces import Dict, Box
        >>> from gymnasium.wrappers import TransformObservation
        >>> from gymnasium.wrappers.vector import VectorizeTransformObservation, FilterObservation
        >>> envs = gym.make_vec("CartPole-v0", num_envs=3)
        >>> make_dict = lambda x: {"obs": x, "junk": np.array([0.0])}
        >>> new_space = Dict({"obs": envs.single_observation_space, "junk": Box(low=-1.0, high=1.0)})
        >>> envs = VectorizeTransformObservation(env=envs, wrapper=TransformObservation, func=make_dict, observation_space=new_space)
        >>> envs = FilterObservation(envs, ["obs"])
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs
        >>> OrderedDict([('obs', array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598],
               [ 0.03517495, -0.000635  , -0.01098382, -0.03203924]],
              dtype=float32))])
    """

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
    """Observation wrapper that flattens the observation.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CarRacing-v2", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.shape
        (3, 96, 96, 3)
        >>> envs = gym.make_vec("CarRacing-v2", num_envs=3)
        >>> envs = FlattenObservation(envs)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.shape
        (3, 27648)
    """

    def __init__(self, env: VectorEnv):
        """Constructor for any environment's observation space that implements ``spaces.utils.flatten_space`` and ``spaces.utils.flatten``.

        Args:
            env:  The vector environment to wrap
        """
        super().__init__(env, transform_observation.FlattenObservation)


class GrayscaleObservation(VectorizeTransformObservation):
    """Observation wrapper that converts an RGB image to grayscale.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CarRacing-v2", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.shape
        (3, 96, 96, 3)
        >>> envs = gym.make_vec("CarRacing-v2", num_envs=3)
        >>> envs = GrayscaleObservation(envs)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.shape
        (3, 96, 96)
    """

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
    """Resizes image observations using OpenCV to shape.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CarRacing-v2", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.shape
        (3, 96, 96, 3)
        >>> envs = gym.make_vec("CarRacing-v2", num_envs=3)
        >>> envs = ResizeObservation(envs, shape=(28, 28))
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.shape
        (3, 28, 28, 3)
    """

    def __init__(self, env: VectorEnv, shape: tuple[int, ...]):
        """Constructor that requires an image environment observation space with a shape.

        Args:
            env: The vector environment to wrap
            shape: The resized observation shape
        """
        super().__init__(env, transform_observation.ResizeObservation, shape=shape)


class ReshapeObservation(VectorizeTransformObservation):
    """Reshapes array based observations to shapes.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CarRacing-v2", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.shape
        (3, 96, 96, 3)
        >>> envs = gym.make_vec("CarRacing-v2", num_envs=3)
        >>> envs = ReshapeObservation(envs, shape=(9216, 3))
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.shape
        (3, 9216, 3)
    """

    def __init__(self, env: VectorEnv, shape: int | tuple[int, ...]):
        """Constructor for env with Box observation space that has a shape product equal to the new shape product.

        Args:
            env: The vector environment to wrap
            shape: The reshaped observation space
        """
        super().__init__(env, transform_observation.ReshapeObservation, shape=shape)


class RescaleObservation(VectorizeTransformObservation):
    """Linearly rescales observation to between a minimum and maximum value.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v0", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.min()
        -0.0446179
        >>> obs.max()
        0.0469136
        >>> envs = gym.make_vec("CartPole-v0", num_envs=3)
        >>> envs = RescaleObservation(envs, min_obs=-5.0, max_obs=5.0)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.min()
        -0.33379582
        >>> obs.max()
        0.55998987
    """

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
    """Observation wrapper for transforming the dtype of an observation.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v0", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.dtype
        dtype('float32')
        >>> envs = gym.make_vec("CartPole-v0", num_envs=3)
        >>> envs = DtypeObservation(envs, dtype=np.int32)
        >>> obs, info = envs.reset(seed=123)
        >>> envs.close()
        >>> obs.dtype
        dtype('int32')
    """

    def __init__(self, env: VectorEnv, dtype: Any):
        """Constructor for Dtype observation wrapper.

        Args:
            env: The vector environment to wrap
            dtype: The new dtype of the observation
        """
        super().__init__(env, transform_observation.DtypeObservation, dtype=dtype)
