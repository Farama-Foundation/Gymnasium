"""Functional to Environment compatibility."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrng

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from gymnasium.experimental.functional import ActType, FuncEnv, StateType
from gymnasium.utils import seeding
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space


class FunctionalJaxEnv(gym.Env):
    """A conversion layer for jax-based environments."""

    state: StateType
    rng: jrng.PRNGKey

    def __init__(
        self,
        func_env: FuncEnv,
        metadata: dict[str, Any] | None = None,
        render_mode: str | None = None,
        spec: EnvSpec | None = None,
    ):
        """Initialize the environment from a FuncEnv."""
        if metadata is None:
            # metadata.get("jax", False) can be used downstream to know that the environment returns jax arrays
            metadata = {"render_mode": [], "jax": True}

        self.func_env = func_env

        self.observation_space = func_env.observation_space
        self.action_space = func_env.action_space

        self.metadata = metadata
        self.render_mode = render_mode

        self.spec = spec

        if self.render_mode == "rgb_array":
            self.render_state = self.func_env.render_init()
        else:
            self.render_state = None

        np_random, _ = seeding.np_random()
        seed = np_random.integers(0, 2**32 - 1, dtype="uint32")

        self.rng = jrng.PRNGKey(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment using the seed."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = jrng.PRNGKey(seed)

        rng, self.rng = jrng.split(self.rng)

        self.state = self.func_env.initial(rng=rng)
        obs = self.func_env.observation(self.state, rng)
        info = self.func_env.state_info(self.state)

        return obs, info

    def step(self, action: ActType):
        """Steps through the environment using the action."""
        rng, self.rng = jrng.split(self.rng)

        next_state = self.func_env.transition(self.state, action, rng)
        observation = self.func_env.observation(next_state, rng)
        reward = self.func_env.reward(self.state, action, next_state, rng)
        terminated = self.func_env.terminal(next_state, rng)
        info = self.func_env.transition_info(self.state, action, next_state)
        self.state = next_state

        return observation, float(reward), bool(terminated), False, info

    def render(self):
        """Returns the render state if `render_mode` is "rgb_array"."""
        if self.render_mode == "rgb_array":
            self.render_state, image = self.func_env.render_image(
                self.state, self.render_state
            )
            return image
        else:
            raise NotImplementedError

    def close(self):
        """Closes the environments and render state if set."""
        if self.render_state is not None:
            self.func_env.render_close(self.render_state)
            self.render_state = None


class FunctionalJaxVectorEnv(gym.vector.VectorEnv):
    """A vector env implementation for functional Jax envs."""

    state: StateType
    rng: jrng.PRNGKey

    def __init__(
        self,
        func_env: FuncEnv,
        num_envs: int,
        max_episode_steps: int = 0,
        metadata: dict[str, Any] | None = None,
        render_mode: str | None = None,
        spec: EnvSpec | None = None,
    ):
        """Initialize the environment from a FuncEnv."""
        super().__init__()
        if metadata is None:
            metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}
        self.func_env = func_env
        self.num_envs = num_envs

        self.single_observation_space = func_env.observation_space
        self.single_action_space = func_env.action_space
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.metadata = metadata
        self.render_mode = render_mode
        self.spec = spec
        self.time_limit = max_episode_steps

        self.steps = jnp.zeros(self.num_envs, dtype=jnp.int32)

        self.prev_done = jnp.zeros(self.num_envs, dtype=jnp.bool_)

        if self.render_mode == "rgb_array":
            self.render_state = self.func_env.render_init()
        else:
            self.render_state = None

        np_random, _ = seeding.np_random()
        seed = np_random.integers(0, 2**32 - 1, dtype="uint32")

        self.rng = jrng.PRNGKey(seed)

        self.func_env.transform(jax.vmap)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = jrng.PRNGKey(seed)

        rng, self.rng = jrng.split(self.rng)

        rng = jrng.split(rng, self.num_envs)

        self.state = self.func_env.initial(rng=rng)
        obs = self.func_env.observation(self.state, rng)
        info = self.func_env.state_info(self.state)

        self.steps = jnp.zeros(self.num_envs, dtype=jnp.int32)

        return obs, info

    def step(self, action: ActType):
        """Steps through the environment using the action."""
        self.steps += 1

        rng, self.rng = jrng.split(self.rng)

        rng = jrng.split(rng, self.num_envs)

        next_state = self.func_env.transition(self.state, action, rng)
        reward = self.func_env.reward(self.state, action, next_state, rng)

        terminated = self.func_env.terminal(next_state, rng)
        truncated = (
            self.steps >= self.time_limit
            if self.time_limit > 0
            else jnp.zeros_like(terminated)
        )

        info = self.func_env.transition_info(self.state, action, next_state)

        if jnp.any(self.prev_done):
            to_reset = jnp.where(self.prev_done)[0]
            reset_count = to_reset.shape[0]

            rng, self.rng = jrng.split(self.rng)
            rng = jrng.split(rng, reset_count)

            new_initials = self.func_env.initial(rng)

            next_state = self.state.at[to_reset].set(new_initials)
            self.steps = self.steps.at[to_reset].set(0)
            terminated = terminated.at[to_reset].set(False)
            truncated = truncated.at[to_reset].set(False)

        self.prev_done = jnp.logical_or(terminated, truncated)

        rng = jrng.split(self.rng, self.num_envs)

        observation = self.func_env.observation(next_state, rng)

        self.state = next_state

        return observation, reward, terminated, truncated, info

    def render(self):
        """Returns the render state if `render_mode` is "rgb_array"."""
        if self.render_mode == "rgb_array":
            self.render_state, image = self.func_env.render_image(
                self.state, self.render_state
            )
            return image
        else:
            raise NotImplementedError

    def close(self):
        """Closes the environments and render state if set."""
        if self.render_state is not None:
            self.func_env.render_close(self.render_state)
            self.render_state = None
