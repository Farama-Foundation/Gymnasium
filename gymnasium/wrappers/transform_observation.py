"""A collection of observation wrappers using a lambda function.

* ``TransformObservation`` - Transforms the observation with a function
* ``FilterObservation`` - Filters a ``Tuple`` or ``Dict`` to only include certain keys
* ``FlattenObservation`` - Flattens the observations
* ``GrayscaleObservation`` - Converts a RGB observation to a grayscale observation
* ``ResizeObservation`` - Resizes an array-based observation (normally a RGB observation)
* ``ReshapeObservation`` - Reshapes an array-based observation
* ``RescaleObservation`` - Rescales an observation to between a minimum and maximum value
* ``DtypeObservation`` - Convert an observation to a dtype
* ``RenderObservation`` - Allows the observation to the rendered frame
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.error import DependencyNotInstalled


__all__ = [
    "TransformObservation",
    "FilterObservation",
    "FlattenObservation",
    "GrayscaleObservation",
    "ResizeObservation",
    "ReshapeObservation",
    "RescaleObservation",
    "DtypeObservation",
    "AddRenderObservation",
]

from gymnasium.wrappers.utils import rescale_box


class TransformObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Applies a function to the ``observation`` received from the environment's :meth:`Env.reset` and :meth:`Env.step` that is passed back to the user.

    The function :attr:`func` will be applied to all observations.
    If the observations from :attr:`func` are outside the bounds of the ``env``'s observation space, provide an updated :attr:`observation_space`.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.TransformObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformObservation
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = gym.make("CartPole-v1")
        >>> env.reset(seed=42)
        (array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), {})
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.random(obs.shape), env.observation_space)
        >>> env.reset(seed=42)
        (array([0.08227695, 0.06540678, 0.09613613, 0.07422512]), {})

    Change logs:
     * v0.15.4 - Initially added
     * v1.0.0 - Add requirement of ``observation_space``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[ObsType], Any],
        observation_space: gym.Space[WrapperObsType] | None,
    ):
        """Constructor for the transform observation wrapper.

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


class FilterObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Filters a Dict or Tuple observation spaces by a set of keys or indexes.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.FilterObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FilterObservation
        >>> env = gym.make("CartPole-v1")
        >>> env = gym.wrappers.TimeAwareObservation(env, flatten=False)
        >>> env.observation_space
        Dict('obs': Box([-4.8               -inf -0.41887903        -inf], [4.8               inf 0.41887903        inf], (4,), float32), 'time': Box(0, 500, (1,), int32))
        >>> env.reset(seed=42)
        ({'obs': array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), 'time': array([0], dtype=int32)}, {})
        >>> env = FilterObservation(env, filter_keys=['time'])
        >>> env.reset(seed=42)
        ({'time': array([0], dtype=int32)}, {})
        >>> env.step(0)
        ({'time': array([1], dtype=int32)}, 1.0, False, False, {})

    Change logs:
     * v0.12.3 - Initially added, originally called `FilterObservationWrapper`
     * v1.0.0 - Rename to `FilterObservation` and add support for tuple observation spaces with integer ``filter_keys``
    """

    def __init__(
        self, env: gym.Env[ObsType, ActType], filter_keys: Sequence[str | int]
    ):
        """Constructor for the filter observation wrapper.

        Args:
            env: The environment to wrap
            filter_keys: The set of subspaces to be *included*, use a list of strings for ``Dict`` and integers for ``Tuple`` spaces
        """
        if not isinstance(filter_keys, Sequence):
            raise TypeError(
                f"Expects `filter_keys` to be a Sequence, actual type: {type(filter_keys)}"
            )
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
                    "The observation space is empty due to filtering all of the keys."
                )

            TransformObservation.__init__(
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

            TransformObservation.__init__(
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


class FlattenObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Flattens the environment's observation space and each observation from ``reset`` and ``step`` functions.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.FlattenObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FlattenObservation
        >>> env = gym.make("CarRacing-v3")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (27648,)

    Change logs:
     * v0.15.0 - Initially added
    """

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for any environment's observation space that implements ``spaces.utils.flatten_space`` and ``spaces.utils.flatten``.

        Args:
            env:  The environment to wrap
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        TransformObservation.__init__(
            self,
            env=env,
            func=lambda obs: spaces.utils.flatten(env.observation_space, obs),
            observation_space=spaces.utils.flatten_space(env.observation_space),
        )


class GrayscaleObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Converts an image observation computed by ``reset`` and ``step`` from RGB to Grayscale.

    The :attr:`keep_dim` will keep the channel dimension.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.GrayscaleObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import GrayscaleObservation
        >>> env = gym.make("CarRacing-v3")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> grayscale_env = GrayscaleObservation(env)
        >>> grayscale_env.observation_space.shape
        (96, 96)
        >>> grayscale_env = GrayscaleObservation(env, keep_dim=True)
        >>> grayscale_env.observation_space.shape
        (96, 96, 1)

    Change logs:
     * v0.15.0 - Initially added, originally called ``GrayScaleObservation``
     * v1.0.0 - Renamed to ``GrayscaleObservation``
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
            TransformObservation.__init__(
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
            TransformObservation.__init__(
                self,
                env=env,
                func=lambda obs: np.sum(
                    np.multiply(obs, np.array([0.2125, 0.7154, 0.0721])), axis=-1
                ).astype(np.uint8),
                observation_space=new_observation_space,
            )


class ResizeObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Resizes image observations using OpenCV to a specified shape.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.ResizeObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ResizeObservation
        >>> env = gym.make("CarRacing-v3")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> resized_env = ResizeObservation(env, (32, 32))
        >>> resized_env.observation_space.shape
        (32, 32, 3)

    Change logs:
     * v0.12.6 - Initially added
     * v1.0.0 - Requires ``shape`` with a tuple of two integers
    """

    def __init__(self, env: gym.Env[ObsType, ActType], shape: tuple[int, int]):
        """Constructor that requires an image environment observation space with a shape.

        Args:
            env: The environment to wrap
            shape: The resized observation shape
        """
        assert isinstance(env.observation_space, spaces.Box)
        assert len(env.observation_space.shape) in {2, 3}
        assert np.all(env.observation_space.low == 0) and np.all(
            env.observation_space.high == 255
        )
        assert env.observation_space.dtype == np.uint8

        assert isinstance(shape, tuple)
        assert len(shape) == 2
        assert all(np.issubdtype(type(elem), np.integer) for elem in shape)
        assert all(x > 0 for x in shape)

        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                'opencv (cv2) is not installed, run `pip install "gymnasium[other]"`'
            ) from e

        self.shape: Final[tuple[int, int]] = tuple(shape)
        # for some reason, cv2.resize will return the shape in reverse, todo confirm implementation
        self.cv2_shape: Final[tuple[int, int]] = (shape[1], shape[0])

        new_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.shape + env.observation_space.shape[2:],
            dtype=np.uint8,
        )

        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        TransformObservation.__init__(
            self,
            env=env,
            func=lambda obs: cv2.resize(
                obs, self.cv2_shape, interpolation=cv2.INTER_AREA
            ),
            observation_space=new_observation_space,
        )


class ReshapeObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Reshapes Array based observations to a specified shape.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.RescaleObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ReshapeObservation
        >>> env = gym.make("CarRacing-v3")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> reshape_env = ReshapeObservation(env, (24, 4, 96, 1, 3))
        >>> reshape_env.observation_space.shape
        (24, 4, 96, 1, 3)

    Change logs:
     * v1.0.0 - Initially added
    """

    def __init__(self, env: gym.Env[ObsType, ActType], shape: int | tuple[int, ...]):
        """Constructor for env with ``Box`` observation space that has a shape product equal to the new shape product.

        Args:
            env: The environment to wrap
            shape: The reshaped observation space
        """
        assert isinstance(env.observation_space, spaces.Box)
        assert np.prod(shape) == np.prod(env.observation_space.shape)

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
        TransformObservation.__init__(
            self,
            env=env,
            func=lambda obs: np.reshape(obs, shape),
            observation_space=new_observation_space,
        )


class RescaleObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Affinely (linearly) rescales a ``Box`` observation space of the environment to within the range of ``[min_obs, max_obs]``.

    For unbounded components in the original observation space, the corresponding target bounds must also be infinite and vice versa.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.RescaleObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import RescaleObservation
        >>> env = gym.make("Pendulum-v1")
        >>> env.observation_space
        Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
        >>> env = RescaleObservation(env, np.array([-2, -1, -10], dtype=np.float32), np.array([1, 0, 1], dtype=np.float32))
        >>> env.observation_space
        Box([ -2.  -1. -10.], [1. 0. 1.], (3,), float32)

    Change logs:
     * v1.0.0 - Initially added
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

        gym.utils.RecordConstructorArgs.__init__(self, min_obs=min_obs, max_obs=max_obs)

        obs_space, func, _ = rescale_box(env.observation_space, min_obs, max_obs)
        TransformObservation.__init__(
            self,
            env=env,
            func=func,
            observation_space=obs_space,
        )


class DtypeObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Modifies the dtype of an observation array to a specified dtype.

    Note:
        This is only compatible with :class:`Box`, :class:`Discrete`, :class:`MultiDiscrete` and :class:`MultiBinary` observation spaces

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.DtypeObservation`.

    Change logs:
     * v1.0.0 - Initially added
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
        TransformObservation.__init__(
            self,
            env=env,
            func=lambda obs: dtype(obs),
            observation_space=new_observation_space,
        )


class AddRenderObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Includes the rendered observations in the environment's observations.

    Notes:
       This was previously called ``PixelObservationWrapper``.

    No vector version of the wrapper exists.

    Example - Replace the observation with the rendered image:
        >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
        >>> env = AddRenderObservation(env, render_only=True)
        >>> env.observation_space
        Box(0, 255, (400, 600, 3), uint8)
        >>> obs, _ = env.reset(seed=123)
        >>> image = env.render()
        >>> np.all(obs == image)
        np.True_
        >>> obs, *_ = env.step(env.action_space.sample())
        >>> image = env.render()
        >>> np.all(obs == image)
        np.True_

    Example - Add the rendered image to the original observation as a dictionary item:
        >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
        >>> env = AddRenderObservation(env, render_only=False)
        >>> env.observation_space
        Dict('pixels': Box(0, 255, (400, 600, 3), uint8), 'state': Box([-4.8               -inf -0.41887903        -inf], [4.8               inf 0.41887903        inf], (4,), float32))
        >>> obs, info = env.reset(seed=123)
        >>> obs.keys()
        dict_keys(['state', 'pixels'])
        >>> obs["state"]
        array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32)
        >>> np.all(obs["pixels"] == env.render())
        np.True_
        >>> obs, reward, terminates, truncates, info = env.step(env.action_space.sample())
        >>> image = env.render()
        >>> np.all(obs["pixels"] == image)
        np.True_

    Change logs:
     * v0.15.0 - Initially added as ``PixelObservationWrapper``
     * v1.0.0 - Renamed to ``AddRenderObservation``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        render_only: bool = True,
        render_key: str = "pixels",
        obs_key: str = "state",
    ):
        """Constructor of the add render observation wrapper.

        Args:
            env: The environment to wrap.
            render_only (bool): If ``True`` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If ``False``, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_key: Optional custom string specifying the pixel key. Defaults to "pixels"
            obs_key: Optional custom string specifying the obs key. Defaults to "state"
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            pixels_only=render_only,
            pixels_key=render_key,
            obs_key=obs_key,
        )

        assert env.render_mode is not None and env.render_mode != "human"
        env.reset()
        pixels = env.render()
        assert pixels is not None and isinstance(pixels, np.ndarray)
        pixel_space = spaces.Box(low=0, high=255, shape=pixels.shape, dtype=np.uint8)

        if render_only:
            obs_space = pixel_space
            TransformObservation.__init__(
                self, env=env, func=lambda _: self.render(), observation_space=obs_space
            )
        elif isinstance(env.observation_space, spaces.Dict):
            assert render_key not in env.observation_space.spaces.keys()

            obs_space = spaces.Dict(
                {render_key: pixel_space, **env.observation_space.spaces}
            )
            TransformObservation.__init__(
                self,
                env=env,
                func=lambda obs: {render_key: self.render(), **obs},
                observation_space=obs_space,
            )
        else:
            obs_space = spaces.Dict(
                {obs_key: env.observation_space, render_key: pixel_space}
            )
            TransformObservation.__init__(
                self,
                env=env,
                func=lambda obs: {obs_key: obs, render_key: self.render()},
                observation_space=obs_space,
            )
