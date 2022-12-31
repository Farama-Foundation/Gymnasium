"""Wrapper for resizing observations."""
from typing import Union

import numpy as np

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Box


class ResizeObservation(gym.ObservationWrapper):
    """Resize the image observation.

    This wrapper works on environments with image observations (or more generally observations of shape AxBxC) and resizes
    the observation to the shape given by the 2-tuple :attr:`shape`. The argument :attr:`shape` may also be an integer.
    In that case, the observation is scaled to a square of side-length :attr:`shape`.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = ResizeObservation(env, 64)
        >>> env.observation_space.shape
        (64, 64, 3)
    """

    def __init__(self, env: gym.Env, shape: Union[tuple[int, int], int]) -> None:
        """Resizes image observations to shape given by :attr:`shape`.

        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        if not isinstance(env.observation_space, Box):
            raise ValueError(f"Expected the observation space to be Box, actual type: {type(env.observation_space)}")
        dims = len(env.observation_space.shape)
        if not 2 <= dims <= 3:
            raise ValueError(f"Expected the observation space to have 2 or 3 dimensions, got: {dims}")

        try:
            shape = tuple(shape)
        except TypeError:
            shape = (shape, shape)
        if len(shape) != 2 or not all(isinstance(x, int) and x > 0 for x in shape):
            raise ValueError(f"Expected shape to be a 2-tuple of positive integers, got: {shape}")

        super().__init__(env)
        self.shape = tuple(shape)

        obs_shape = self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                "opencv is not installed, run `pip install gymnasium[other]`"
            ) from e

        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        return observation.reshape(self.observation_space.shape)
