"""A collection of observation wrappers using a lambda function.

* ``LambdaObservationV0`` - Transforms the observation with a function
* ``FilterObservationV0`` - Filters a ``Tuple`` or ``Dict`` to only include certain keys
* ``FlattenObservationV0`` - Flattens the observations
* ``GrayscaleObservationV0`` - Converts a RGB observation to a grayscale observation
* ``ResizeObservationV0`` - Resizes an array-based observation (normally a RGB observation)
* ``ReshapeObservationV0`` - Reshapes an array-based observation
* ``RescaleObservationV0`` - Rescales an observation to between a minimum and maximum value
* ``DtypeObservationV0`` - Convert an observation to a dtype
* ``PixelObservationV0`` - Allows the observation to the rendered frame
"""
from __future__ import annotations

from typing import Any, Callable, Final, Sequence

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.error import DependencyNotInstalled


__all__ = [
    "LambdaObservationV0",
    "FilterObservationV0",
    "FlattenObservationV0",
    "GrayscaleObservationV0",
    "ResizeObservationV0",
    "ReshapeObservationV0",
    "RescaleObservationV0",
    "DtypeObservationV0",
    "PixelObservationV0",
]


class LambdaObservationV0(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Transforms an observation via a function provided to the wrapper.

    The function :attr:`func` will be applied to all observations.
    If the observations from :attr:`func` are outside the bounds of the ``env``'s observation space, provide an :attr:`observation_space`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import LambdaObservationV0
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = gym.make("CartPole-v1")
        >>> env = LambdaObservationV0(env, lambda obs: obs + 0.1 * np.random.random(obs.shape), env.observation_space)
        >>> env.reset(seed=42)
        (array([0.08227695, 0.06540678, 0.09613613, 0.07422512]), {})
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[ObsType], Any],
        observation_space: gym.Space[WrapperObsType] | None,
    ):
        """Constructor for the lambda observation wrapper.

        Args:
            env: The environment to wrap
            func: A function that will transform an observation. If this transformed observation is outside the observation space of ``env.observation_space`` then provide an `observation_space`.
            observation_space: The observation spaces of the wrapper, if None, then it is assumed the same as ``env.observation_space``.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, func=func, observation_space=observation_space
        )
        gym.ObservationWrapper.__init__(self, env)

        if observation_space is not None:
            self.observation_space = observation_space

        self.func = func

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return self.func(observation)


class FilterObservationV0(
    LambdaObservationV0[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Filters Dict or Tuple observation space by the keys or indexes.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformObservation
        >>> from gymnasium.experimental.wrappers import FilterObservationV0
        >>> env = gym.make("CartPole-v1")
        >>> env = gym.wrappers.TransformObservation(env, lambda obs: {'obs': obs, 'time': 0})
        >>> env.observation_space = gym.spaces.Dict(obs=env.observation_space, time=gym.spaces.Discrete(1))
        >>> env.reset(seed=42)
        ({'obs': array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), 'time': 0}, {})
        >>> env = FilterObservationV0(env, filter_keys=['time'])
        >>> env.reset(seed=42)
        ({'time': 0}, {})
        >>> env.step(0)
        ({'time': 0}, 1.0, False, False, {})
    """

    def __init__(
        self, env: gym.Env[ObsType, ActType], filter_keys: Sequence[str | int]
    ):
        """Constructor for the filter observation wrapper.

        Args:
            env: The environment to wrap
            filter_keys: The subspaces to be included, use a list of strings or integers for ``Dict`` and ``Tuple`` spaces respectivesly
        """
        assert isinstance(filter_keys, Sequence)
        gym.utils.RecordConstructorArgs.__init__(self, filter_keys=filter_keys)

        # Filters for dictionary space
        if isinstance(env.observation_space, spaces.Dict):
            assert all(isinstance(key, str) for key in filter_keys)

            if any(
                key not in env.observation_space.spaces.keys() for key in filter_keys
            ):
                missing_keys = [
                    key
                    for key in filter_keys
                    if key not in env.observation_space.spaces.keys()
                ]
                raise ValueError(
                    "All the `filter_keys` must be included in the observation space.\n"
                    f"Filter keys: {filter_keys}\n"
                    f"Observation keys: {list(env.observation_space.spaces.keys())}\n"
                    f"Missing keys: {missing_keys}"
                )

            new_observation_space = spaces.Dict(
                {key: env.observation_space[key] for key in filter_keys}
            )
            if len(new_observation_space) == 0:
                raise ValueError(
                    "The observation space is empty due to filtering all keys."
                )

            LambdaObservationV0.__init__(
                self,
                env=env,
                func=lambda obs: {key: obs[key] for key in filter_keys},
                observation_space=new_observation_space,
            )
            # Filter for tuple observation
        elif isinstance(env.observation_space, spaces.Tuple):
            assert all(isinstance(key, int) for key in filter_keys)
            assert len(set(filter_keys)) == len(
                filter_keys
            ), f"Duplicate keys exist, filter_keys: {filter_keys}"

            if any(
                0 < key and key >= len(env.observation_space) for key in filter_keys
            ):
                missing_index = [
                    key
                    for key in filter_keys
                    if 0 < key and key >= len(env.observation_space)
                ]
                raise ValueError(
                    "All the `filter_keys` must be included in the length of the observation space.\n"
                    f"Filter keys: {filter_keys}, length of observation: {len(env.observation_space)}, "
                    f"missing indexes: {missing_index}"
                )

            new_observation_spaces = spaces.Tuple(
                env.observation_space[key] for key in filter_keys
            )
            if len(new_observation_spaces) == 0:
                raise ValueError(
                    "The observation space is empty due to filtering all keys."
                )

            LambdaObservationV0.__init__(
                self,
                env=env,
                func=lambda obs: tuple(obs[key] for key in filter_keys),
                observation_space=new_observation_spaces,
            )
        else:
            raise ValueError(
                f"FilterObservation wrapper is only usable with `Dict` and `Tuple` observations, actual type: {type(env.observation_space)}"
            )

        self.filter_keys: Final[Sequence[str | int]] = filter_keys


class FlattenObservationV0(
    LambdaObservationV0[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Observation wrapper that flattens the observation.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import FlattenObservationV0
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservationV0(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (27648,)
    """

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for any environment's observation space that implements ``spaces.utils.flatten_space`` and ``spaces.utils.flatten``.

        Args:
            env:  The environment to wrap
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        LambdaObservationV0.__init__(
            self,
            env=env,
            func=lambda obs: spaces.utils.flatten(env.observation_space, obs),
            observation_space=spaces.utils.flatten_space(env.observation_space),
        )


class GrayscaleObservationV0(
    LambdaObservationV0[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Observation wrapper that converts an RGB image to grayscale.

    The :attr:`keep_dim` will keep the channel dimension

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import GrayscaleObservationV0
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> grayscale_env = GrayscaleObservationV0(env)
        >>> grayscale_env.observation_space.shape
        (96, 96)
        >>> grayscale_env = GrayscaleObservationV0(env, keep_dim=True)
        >>> grayscale_env.observation_space.shape
        (96, 96, 1)
    """

    def __init__(self, env: gym.Env[ObsType, ActType], keep_dim: bool = False):
        """Constructor for an RGB image based environments to make the image grayscale.

        Args:
            env: The environment to wrap
            keep_dim: If to keep the channel in the observation, if ``True``, ``obs.shape == 3`` else ``obs.shape == 2``
        """
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
        gym.utils.RecordConstructorArgs.__init__(self, keep_dim=keep_dim)

        self.keep_dim: Final[bool] = keep_dim
        if keep_dim:
            new_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=env.observation_space.shape[:2] + (1,),
                dtype=np.uint8,
            )
            LambdaObservationV0.__init__(
                self,
                env=env,
                func=lambda obs: np.expand_dims(
                    np.sum(
                        np.multiply(obs, np.array([0.2125, 0.7154, 0.0721])), axis=-1
                    ).astype(np.uint8),
                    axis=-1,
                ),
                observation_space=new_observation_space,
            )
        else:
            new_observation_space = spaces.Box(
                low=0, high=255, shape=env.observation_space.shape[:2], dtype=np.uint8
            )
            LambdaObservationV0.__init__(
                self,
                env=env,
                func=lambda obs: np.sum(
                    np.multiply(obs, np.array([0.2125, 0.7154, 0.0721])), axis=-1
                ).astype(np.uint8),
                observation_space=new_observation_space,
            )


class ResizeObservationV0(
    LambdaObservationV0[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Resizes image observations using OpenCV to shape.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import ResizeObservationV0
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> resized_env = ResizeObservationV0(env, (32, 32))
        >>> resized_env.observation_space.shape
        (32, 32, 3)
    """

    def __init__(self, env: gym.Env[ObsType, ActType], shape: tuple[int, ...]):
        """Constructor that requires an image environment observation space with a shape.

        Args:
            env: The environment to wrap
            shape: The resized observation shape
        """
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
        except ImportError as e:
            raise DependencyNotInstalled(
                "opencv (cv2) is not installed, run `pip install gymnasium[other]`"
            ) from e

        self.shape: Final[tuple[int, ...]] = tuple(shape)

        new_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.shape + env.observation_space.shape[2:],
            dtype=np.uint8,
        )

        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        LambdaObservationV0.__init__(
            self,
            env=env,
            func=lambda obs: cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA),
            observation_space=new_observation_space,
        )


class ReshapeObservationV0(
    LambdaObservationV0[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Reshapes array based observations to shapes.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import ReshapeObservationV0
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> reshape_env = ReshapeObservationV0(env, (24, 4, 96, 1, 3))
        >>> reshape_env.observation_space.shape
        (24, 4, 96, 1, 3)
    """

    def __init__(self, env: gym.Env[ObsType, ActType], shape: int | tuple[int, ...]):
        """Constructor for env with ``Box`` observation space that has a shape product equal to the new shape product.

        Args:
            env: The environment to wrap
            shape: The reshaped observation space
        """
        assert isinstance(env.observation_space, spaces.Box)
        assert np.product(shape) == np.product(env.observation_space.shape)

        assert isinstance(shape, tuple)
        assert all(np.issubdtype(type(elem), np.integer) for elem in shape)
        assert all(x > 0 or x == -1 for x in shape)

        new_observation_space = spaces.Box(
            low=np.reshape(np.ravel(env.observation_space.low), shape),
            high=np.reshape(np.ravel(env.observation_space.high), shape),
            shape=shape,
            dtype=env.observation_space.dtype,
        )
        self.shape = shape

        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        LambdaObservationV0.__init__(
            self,
            env=env,
            func=lambda obs: np.reshape(obs, shape),
            observation_space=new_observation_space,
        )


class RescaleObservationV0(
    LambdaObservationV0[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Linearly rescales observation to between a minimum and maximum value.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import RescaleObservationV0
        >>> env = gym.make("Pendulum-v1")
        >>> env.observation_space
        Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
        >>> env = RescaleObservationV0(env, np.array([-2, -1, -10], dtype=np.float32), np.array([1, 0, 1], dtype=np.float32))
        >>> env.observation_space
        Box([ -2.  -1. -10.], [1. 0. 1.], (3,), float32)
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        min_obs: np.floating | np.integer | np.ndarray,
        max_obs: np.floating | np.integer | np.ndarray,
    ):
        """Constructor that requires the env observation spaces to be a :class:`Box`.

        Args:
            env: The environment to wrap
            min_obs: The new minimum observation bound
            max_obs: The new maximum observation bound
        """
        assert isinstance(env.observation_space, spaces.Box)
        assert not np.any(env.observation_space.low == np.inf) and not np.any(
            env.observation_space.high == np.inf
        )

        if not isinstance(min_obs, np.ndarray):
            assert np.issubdtype(type(min_obs), np.integer) or np.issubdtype(
                type(max_obs), np.floating
            )
            min_obs = np.full(env.observation_space.shape, min_obs)
        assert (
            min_obs.shape == env.observation_space.shape
        ), f"{min_obs.shape}, {env.observation_space.shape}, {min_obs}, {env.observation_space.low}"
        assert not np.any(min_obs == np.inf)

        if not isinstance(max_obs, np.ndarray):
            assert np.issubdtype(type(max_obs), np.integer) or np.issubdtype(
                type(max_obs), np.floating
            )
            max_obs = np.full(env.observation_space.shape, max_obs)
        assert max_obs.shape == env.observation_space.shape
        assert not np.any(max_obs == np.inf)

        self.min_obs = min_obs
        self.max_obs = max_obs

        # Imagine the x-axis between the old Box and the y-axis being the new Box
        gradient = (max_obs - min_obs) / (
            env.observation_space.high - env.observation_space.low
        )
        intercept = gradient * -env.observation_space.low + min_obs

        gym.utils.RecordConstructorArgs.__init__(self, min_obs=min_obs, max_obs=max_obs)
        LambdaObservationV0.__init__(
            self,
            env=env,
            func=lambda obs: gradient * obs + intercept,
            observation_space=spaces.Box(
                low=min_obs,
                high=max_obs,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype,
            ),
        )


class DtypeObservationV0(
    LambdaObservationV0[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Observation wrapper for transforming the dtype of an observation.

    Note:
        This is only compatible with :class:`Box`, :class:`Discrete`, :class:`MultiDiscrete` and :class:`MultiBinary` observation spaces
    """

    def __init__(self, env: gym.Env[ObsType, ActType], dtype: Any):
        """Constructor for Dtype observation wrapper.

        Args:
            env: The environment to wrap
            dtype: The new dtype of the observation
        """
        assert isinstance(
            env.observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        )

        self.dtype = dtype
        if isinstance(env.observation_space, spaces.Box):
            new_observation_space = spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                shape=env.observation_space.shape,
                dtype=self.dtype,
            )
        elif isinstance(env.observation_space, spaces.Discrete):
            new_observation_space = spaces.Box(
                low=env.observation_space.start,
                high=env.observation_space.start + env.observation_space.n,
                shape=(),
                dtype=self.dtype,
            )
        elif isinstance(env.observation_space, spaces.MultiDiscrete):
            new_observation_space = spaces.MultiDiscrete(
                env.observation_space.nvec, dtype=dtype
            )
        elif isinstance(env.observation_space, spaces.MultiBinary):
            new_observation_space = spaces.Box(
                low=0,
                high=1,
                shape=env.observation_space.shape,
                dtype=self.dtype,
            )
        else:
            raise TypeError(
                "DtypeObservation is only compatible with value / array-based observations."
            )

        gym.utils.RecordConstructorArgs.__init__(self, dtype=dtype)
        LambdaObservationV0.__init__(
            self,
            env=env,
            func=lambda obs: dtype(obs),
            observation_space=new_observation_space,
        )


class PixelObservationV0(
    LambdaObservationV0[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Includes the rendered observations to the environment's observations.

    Observations of this wrapper will be dictionaries of images.
    You can also choose to add the observation of the base environment to this dictionary.
    In that case, if the base environment has an observation space of type :class:`Dict`, the dictionary
    of rendered images will be updated with the base environment's observation. If, however, the observation
    space is of type :class:`Box`, the base environment's observation (which will be an element of the :class:`Box`
    space) will be added to the dictionary under the key "state".
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        pixels_only: bool = True,
        pixels_key: str = "pixels",
        obs_key: str = "state",
    ):
        """Constructor of the pixel observation wrapper.

        Args:
            env: The environment to wrap.
            pixels_only (bool): If ``True`` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If ``False``, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            pixels_key: Optional custom string specifying the pixel key. Defaults to "pixels"
            obs_key: Optional custom string specifying the obs key. Defaults to "state"
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, pixels_only=pixels_only, pixels_key=pixels_key, obs_key=obs_key
        )

        assert env.render_mode is not None and env.render_mode != "human"
        env.reset()
        pixels = env.render()
        assert pixels is not None and isinstance(pixels, np.ndarray)
        pixel_space = spaces.Box(low=0, high=255, shape=pixels.shape, dtype=np.uint8)

        if pixels_only:
            obs_space = pixel_space
            LambdaObservationV0.__init__(
                self, env=env, func=lambda _: self.render(), observation_space=obs_space
            )
        elif isinstance(env.observation_space, spaces.Dict):
            assert pixels_key not in env.observation_space.spaces.keys()

            obs_space = spaces.Dict(
                {pixels_key: pixel_space, **env.observation_space.spaces}
            )
            LambdaObservationV0.__init__(
                self,
                env=env,
                func=lambda obs: {pixels_key: self.render(), **obs_space},
                observation_space=obs_space,
            )
        else:
            obs_space = spaces.Dict(
                {obs_key: env.observation_space, pixels_key: pixel_space}
            )
            LambdaObservationV0.__init__(
                self,
                env=env,
                func=lambda obs: {obs_key: obs, pixels_key: self.render()},
                observation_space=obs_space,
            )
