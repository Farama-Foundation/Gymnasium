"""Provides a generic testing environment for use in tests with custom reset, step and render functions."""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.vector.vector_env import AutoresetMode


class DefaultTestSpace:
    """Sentinel class to indicate that the default space should be used."""


DEFAULT_SPACE = DefaultTestSpace()


def basic_reset_func(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,
) -> tuple[ObsType, dict]:
    """A basic reset function that will pass the environment check using random actions from the observation space."""
    super(GenericTestEnv, self).reset(seed=seed)
    self.observation_space.seed(self.np_random_seed)
    return self.observation_space.sample(), {"options": options}


def old_reset_func(self) -> ObsType:
    """An old reset function that will pass the environment check using random actions from the observation space."""
    super(GenericTestEnv, self).reset()
    return self.observation_space.sample()


def basic_step_func(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
    """A step function that follows the basic step api that will pass the environment check using random actions from the observation space."""
    return self.observation_space.sample(), 0, False, False, {}


def old_step_func(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
    """A step function that follows the old step api that will pass the environment check using random actions from the observation space."""
    return self.observation_space.sample(), 0, False, {}


def basic_render_func(self):
    """Basic render fn that does nothing."""
    pass


class GenericTestEnv(gym.Env):
    """A generic testing environment for use in testing with modified environments are required."""

    def __init__(
        self,
        action_space: spaces.Space | DefaultTestSpace | None = DEFAULT_SPACE,
        observation_space: spaces.Space | DefaultTestSpace | None = DEFAULT_SPACE,
        reset_func: Callable = basic_reset_func,
        step_func: Callable = basic_step_func,
        render_func: Callable = basic_render_func,
        metadata: dict[str, Any] | None = None,
        render_mode: str | None = None,
        spec: EnvSpec | None = None,
    ):
        """Generic testing environment constructor.

        Args:
            action_space: The environment action space. Use DEFAULT_SPACE for default,
                None for no space (to test error cases), or a custom Space.
            observation_space: The environment observation space. Use DEFAULT_SPACE for default,
                None for no space (to test error cases), or a custom Space.
            reset_func: The environment reset function
            step_func: The environment step function
            render_func: The environment render function
            metadata: The environment metadata
            render_mode: The render mode of the environment
            spec: The environment spec
        """
        if action_space is DEFAULT_SPACE:
            self.action_space = spaces.Box(0, 1, (1,))
        elif action_space is not None:
            self.action_space = action_space
        # If action_space is None, don't set the attribute (for testing error cases)

        if observation_space is DEFAULT_SPACE:
            self.observation_space = spaces.Box(0, 1, (1,))
        elif observation_space is not None:
            self.observation_space = observation_space
        # If observation_space is None, don't set the attribute (for testing error cases)

        if reset_func is not None:
            self.reset = types.MethodType(reset_func, self)
        if step_func is not None:
            self.step = types.MethodType(step_func, self)
        if render_func is not None:
            self.render = types.MethodType(render_func, self)

        if metadata is None:
            self.metadata = {"render_modes": []}
        else:
            self.metadata = metadata

        self.render_mode = render_mode

        if spec is None:
            self.spec = EnvSpec(
                "TestingEnv-v0",
                "tests.testing_env:GenericTestEnv",
                max_episode_steps=100,
            )
        else:
            self.spec = spec

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> ObsType | tuple[ObsType, dict]:
        """Resets the environment."""
        # If you need a default working reset function, use `basic_reset_fn` above
        raise NotImplementedError("TestingEnv reset_fn is not set.")

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict[str, Any]]:
        """Steps through the environment."""
        raise NotImplementedError("TestingEnv step_fn is not set.")

    def render(self):
        """Renders the environment."""
        raise NotImplementedError("testingEnv render_fn is not set.")


def basic_vector_reset_func(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,
) -> tuple[ObsType, dict]:
    """A basic reset function that will pass the environment check using random actions from the observation space."""
    super(GenericTestVectorEnv, self).reset(seed=seed)
    self.observation_space.seed(self.np_random_seed)
    return self.observation_space.sample(), {"options": options}


def basic_vector_step_func(
    self, action: ActType
) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict]:
    """A step function that follows the basic step api that will pass the environment check using random actions from the observation space."""
    obs = self.observation_space.sample()
    rewards = np.zeros(self.num_envs, dtype=np.float64)
    terminations = np.zeros(self.num_envs, dtype=np.bool_)
    truncations = np.zeros(self.num_envs, dtype=np.bool_)
    return obs, rewards, terminations, truncations, {}


def basic_vector_render_func(self):
    """Basic render fn that does nothing."""
    pass


class GenericTestVectorEnv(VectorEnv):
    """A generic testing vector environment similar to GenericTestEnv.

    Some tests cannot use SyncVectorEnv, e.g. when returning non-numpy arrays in the observations.
    In these cases, GenericTestVectorEnv can be used to simulate a vector environment.
    """

    def __init__(
        self,
        num_envs: int = 1,
        action_space: spaces.Space | DefaultTestSpace | None = DEFAULT_SPACE,
        observation_space: spaces.Space | DefaultTestSpace | None = DEFAULT_SPACE,
        reset_func: Callable = basic_vector_reset_func,
        step_func: Callable = basic_vector_step_func,
        render_func: Callable = basic_vector_render_func,
        metadata: dict[str, Any] | None = None,
        render_mode: str | None = None,
        spec: EnvSpec | None = None,
    ):
        """Generic testing vector environment constructor.

        Args:
            num_envs: The number of environments to create
            action_space: The environment action space. Use DEFAULT_SPACE for default,
                None for no space (to test error cases), or a custom Space.
            observation_space: The environment observation space. Use DEFAULT_SPACE for default,
                None for no space (to test error cases), or a custom Space.
            reset_func: The environment reset function
            step_func: The environment step function
            render_func: The environment render function
            metadata: The environment metadata
            render_mode: The render mode of the environment
            spec: The environment spec
        """
        super().__init__()

        self.num_envs = num_envs
        if metadata is None:
            self.metadata = {
                "render_modes": [],
                "autoreset_mode": AutoresetMode.NEXT_STEP,
            }
        else:
            self.metadata = metadata
        self.render_mode = render_mode
        if spec is None:
            self.spec = EnvSpec(
                "TestingVectorEnv-v0",
                "tests.testing_env:GenericTestVectorEnv",
                max_episode_steps=100,
            )
        else:
            self.spec = spec

        # Set the single spaces and create batched spaces
        if action_space is DEFAULT_SPACE:
            self.single_action_space = spaces.Box(0, 1, (1,))
        elif action_space is not None:
            self.single_action_space = action_space
        # If action_space is None, don't set the attribute (for testing error cases)

        if hasattr(self, "single_action_space"):
            self.action_space = batch_space(self.single_action_space, num_envs)

        if observation_space is DEFAULT_SPACE:
            self.single_observation_space = spaces.Box(0, 1, (1,))
        elif observation_space is not None:
            self.single_observation_space = observation_space
        # If observation_space is None, don't set the attribute (for testing error cases)

        if hasattr(self, "single_observation_space"):
            self.observation_space = batch_space(
                self.single_observation_space, num_envs
            )

        # Bind the functions to the instance
        if reset_func is not None:
            self.reset = types.MethodType(reset_func, self)
        if step_func is not None:
            self.step = types.MethodType(step_func, self)
        if render_func is not None:
            self.render = types.MethodType(render_func, self)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[ObsType, dict]:
        """Resets the environment."""
        # If you need a default working reset function, use `basic_vector_reset_fn` above
        raise NotImplementedError("TestingVectorEnv reset_fn is not set.")

    def step(
        self, action: ActType
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Steps through the environment."""
        raise NotImplementedError("TestingVectorEnv step_fn is not set.")

    def render(self) -> tuple[Any, ...] | None:
        """Renders the environment."""
        raise NotImplementedError("TestingVectorEnv render_fn is not set.")
