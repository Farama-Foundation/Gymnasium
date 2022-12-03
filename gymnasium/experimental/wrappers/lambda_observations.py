"""Lambda observation wrappers which apply a function to the observation."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import jumpy as jp
import numpy as np
import numpy.typing as npt

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import utils


class LambdaObservationV0(gym.ObservationWrapper):
    """Transforms an observation via a function provided to the wrapper.

    The function :attr:`func` will be applied to all observations.
    If the observations from :attr:`func` are outside the bounds of the `env` spaces, provide a :attr:`observation_space`.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> env = gym.make('CartPole-v1')
        >>> env = LambdaObservationV0(env, lambda obs: obs + 0.1 * np.random.random(obs.shape))
        >>> env.reset()
        array([-0.08319338,  0.04635121, -0.07394746,  0.20877492])
    """

    def __init__(
        self,
        env: gym.Env,
        func: Callable[[ObsType], Any],
        observation_space: gym.Space | None,
    ):
        """Constructor for the lambda observation wrapper.

        Args:
            env: The environment to wrap
            func: A function that will transform an observation. If this transformed observation is outside the observation space of `env.observation_space` then provide an `observation_space`.
            observation_space: The observation spaces of the wrapper, if None, then it is assumed the same as `env.observation_space`.
        """
        super().__init__(env)
        if observation_space is not None:
            self.observation_space = observation_space

        self.func = func

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return self.func(observation)


class FilterObservationV0(LambdaObservationV0):
    """Filter Dict observation space by the keys.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.wrappers.TransformObservation(
        ...     gym.make('CartPole-v1'), lambda obs: {'obs': obs, 'time': 0}
        ... )
        >>> env.observation_space = gym.spaces.Dict(obs=env.observation_space, time=gym.spaces.Discrete(1))
        >>> env.reset()
        {'obs': array([-0.00067088, -0.01860439,  0.04772898, -0.01911527], dtype=float32), 'time': 0}
        >>> env = FilterObservationV0(env, filter_keys=['time'])
        >>> env.reset()
        {'obs': array([ 0.04560107,  0.04466959, -0.0328232 , -0.02367178], dtype=float32)}
        >>> env.step(0)
        ({'obs': array([ 0.04649447, -0.14996664, -0.03329664,  0.25847703], dtype=float32)}, 1.0, False, {})
    """

    def __init__(self, env: gym.Env, filter_keys: Sequence[str]):
        """Constructor for an environment with a dictionary observation space where all :attr:`filter_keys` are in the observation space keys."""
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError(
                f"FilterObservation wrapper is only usable with dict observations, actual type: {type(env.observation_space)}"
            )

        if any(key not in env.observation_space.keys() for key in filter_keys):
            missing_keys = [
                key for key in filter_keys if key not in env.observation_space.keys()
            ]
            raise ValueError(
                "All the filter_keys must be included in the original observation space.\n"
                f"Filter keys: {filter_keys}\n"
                f"Observation keys: {list(env.observation_space.keys())}\n"
                f"Missing keys: {missing_keys}"
            )

        new_observation_space = spaces.Dict(
            {key: env.observation_space[key] for key in filter_keys}
        )
        super().__init__(
            env,
            lambda obs: {key: obs[key] for key in filter_keys},
            new_observation_space,
        )


class FlattenObservationV0(LambdaObservationV0):
    """Observation wrapper that flattens the observation.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservationV0(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs, info = env.reset()
        >>> obs.shape
        (27648,)
    """

    def __init__(self, env: gym.Env):
        """Constructor for any environment's observation space that implements ``spaces.utils.flatten_space`` and ``spaces.utils.flatten``."""
        flattened_space = utils.flatten_space(env.observation_space)
        super().__init__(
            env, lambda obs: utils.flatten(flattened_space, obs), flattened_space
        )


class GrayscaleObservationV0(LambdaObservationV0):
    """Observation wrapper that converts an RGB image to grayscale.

    The :attr:`keep_dim` will keep the channel dimension

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CarRacing-v1")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> grayscale_env = GrayscaleObservationV0(env)
        >>> grayscale_env.observation_space.shape
        (96, 96)
        >>> grayscale_env = GrayscaleObservationV0(env, keep_dim=True)
        >>> grayscale_env.observation_space.shape
        (96, 96, 1)
    """

    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """Constructor for an RGB image based environments to make the image grayscale."""
        assert isinstance(env.observation_space, spaces.Box)
        assert (
            len(env.observation_space.shape) == 3
            and env.observation_space.shape[-1] == 3
        )
        assert (
            np.all(env.observation_space.low == 0)
            and np.all(env.observation_space.high == 255)
            and env.observation_space.dtype == np.uint8
        )

        self.keep_dim = keep_dim
        if keep_dim:
            new_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=env.observation_space.shape[:2] + (1,),
                dtype=np.uint8,
            )
            super().__init__(
                env,
                lambda obs: jp.expand_dims(
                    jp.sum(
                        jp.multiply(obs, jp.array([0.2125, 0.7154, 0.0721])), axis=-1
                    )
                ),
                new_observation_space,
            )
        else:
            new_observation_space = spaces.Box(
                low=0, high=255, shape=env.observation_space.shape[:2], dtype=np.uint8
            )
            super().__init__(
                env,
                lambda obs: jp.sum(
                    jp.multiply(obs, jp.array([0.2125, 0.7154, 0.0721])), axis=-1
                ),
                new_observation_space,
            )


class ResizeObservationV0(LambdaObservationV0):
    """Observation wrapper for resize image observations using opencv.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CarRacing-v1")
        >>> resized_env = ResizeObservationV0(env, (32, 32))
        >>> resized_env.observation_space.shape
        (32, 32, 3)
    """

    def __init__(self, env: gym.Env, shape: tuple[int, ...]):
        """Constructor that requires an image environment observation space with a shape."""
        assert isinstance(env.observation_space, spaces.Box)
        assert len(env.observation_space.shape) in [2, 3]
        assert np.all(env.observation_space.low == 0) and np.all(
            env.observation_space.high == 255
        )
        assert env.observation_space.dtype == np.uint8

        assert isinstance(shape, tuple)
        assert all(np.issubdtype(type(elem), np.integer) for elem in shape)
        assert all(x > 0 for x in shape)

        try:
            import cv2
        except ImportError:
            raise DependencyNotInstalled(
                "opencv is not install, run `pip install gymnasium[other]`"
            )

        self.shape = tuple(shape)

        new_observation_space = spaces.Box(
            low=0, high=255, shape=self.shape + env.observation_space.shape[2:]
        )
        super().__init__(
            env,
            lambda obs: cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA),
            new_observation_space,
        )


class ReshapeObservationV0(LambdaObservationV0):
    """Observation wrapper for reshaping the observation."""

    def __init__(self, env: gym.Env, shape: int | tuple[int, ...]):
        """Constructor for env with Box observation space that has a shape product equal to the new shape product."""
        assert isinstance(env.observation_space, spaces.Box)
        assert np.product(shape) == np.product(env.observation_space.shape)

        assert isinstance(shape, tuple)
        assert all(np.issubdtype(type(elem), np.integer) for elem in shape)
        assert all(x > 0 for x in shape)

        new_observation_space = spaces.Box(
            low=np.reshape(np.ravel(env.observation_space.low), shape),
            high=np.reshape(np.ravel(env.observation_space.high), shape),
            shape=shape,
            dtype=env.observation_space.dtype,
        )
        super().__init__(
            env, lambda obs: jp.reshape(obs, self.shape), new_observation_space
        )


class RescaleObservationV0(LambdaObservationV0):
    """Observation wrapper for rescaling the observations between a minimum and maximum value."""

    def __init__(
        self,
        env: gym.Env,
        min_obs: tuple[np.floating, np.integer, np.ndarray],
        max_obs: tuple[np.floating, np.integer, np.ndarray],
    ):
        """Constructor that requires the env observation spaces to be a :class:`Box`."""
        assert isinstance(env.observation_space, spaces.Box)

        if not isinstance(min_obs, np.ndarray):
            assert np.issubdtype(type(min_obs), np.integer) or np.issubdtype(
                type(max_obs), np.floating
            )
            min_obs = np.full(env.observation_space.shape, min_obs)
        assert min_obs.shape == env.observation_space.shape
        assert not np.any(min_obs == np.inf)

        if not isinstance(max_obs, np.ndarray):
            assert np.issubdtype(type(max_obs), np.integer) or np.issubdtype(
                type(max_obs), np.floating
            )
            max_obs = np.full(env.observation_space.shape, max_obs)
        assert max_obs.shape == env.observation_space.shape
        assert not np.any(max_obs == np.inf)

        env_low = env.observation_space.low
        env_high = env.observation_space.high

        new_observation_space = spaces.Box(low=min_obs, high=max_obs)
        super().__init__(
            env,
            lambda obs: env_low
            + (env_high - env_low) * ((obs - min_obs) / (max_obs - min_obs)),
            new_observation_space,
        )


class DtypeObservationV0(LambdaObservationV0):
    """Observation wrapper for transforming the dtype of an observation."""

    def __init__(self, env: gym.Env, dtype: npt.DTypeLike):
        """Constructor for Dtype, this is only valid with :class:`Box`, :class:`Discrete`, :class:`MultiDiscrete` and :class:`MultiBinary` observation spaces."""
        assert isinstance(
            env.observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        )

        dtype = np.dtype(dtype)
        if isinstance(env.observation_space, spaces.Box):
            new_observation_space = spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                shape=env.observation_space.shape,
                dtype=dtype.__name__,
            )
        elif isinstance(env.observation_space, spaces.Discrete):
            new_observation_space = spaces.Box(
                low=env.observation_space.start,
                high=env.observation_space.start + env.observation_space.n,
                shape=(),
                dtype=dtype.__name__,
            )
        elif isinstance(env.observation_space, spaces.MultiDiscrete):
            new_observation_space = spaces.MultiDiscrete(
                env.observation_space.nvec, dtype=dtype.__name__
            )
        elif isinstance(env.observation_space, spaces.MultiBinary):
            new_observation_space = spaces.Box(
                low=0, high=1, shape=env.observation_space.shape, dtype=dtype.__name__
            )
        else:
            raise TypeError

        super().__init__(env, lambda obs: dtype(obs), new_observation_space)
