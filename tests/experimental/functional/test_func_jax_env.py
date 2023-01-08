"""Test the functional jax environment."""

import jax
import jax.numpy as jnp
import jax.random as jrng
import pytest

from gymnasium.envs.phys2d.cartpole import CartPoleFunctional
from gymnasium.envs.phys2d.pendulum import PendulumFunctional


@pytest.mark.parametrize("env_class", [CartPoleFunctional, PendulumFunctional])
def test_without_transform(env_class):
    """Tests the environment without transforming the environment."""
    env = env_class()
    rng = jrng.PRNGKey(0)

    state = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        action = env.action_space.sample()
        next_state = env.transition(state, action, None)
        reward = env.reward(state, action, next_state)
        terminal = env.terminal(next_state)

        assert next_state.shape == state.shape
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        assert next_state.dtype == jnp.float32
        assert isinstance(obs, jnp.ndarray)
        assert obs.dtype == jnp.float32

        state = next_state


@pytest.mark.parametrize("env_class", [CartPoleFunctional, PendulumFunctional])
def test_jit(env_class):
    """Tests jitting the functional instance functions."""
    env = env_class()
    rng = jrng.PRNGKey(0)

    env.transform(jax.jit)
    state = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        action = env.action_space.sample()
        next_state = env.transition(state, action, None)
        reward = env.reward(state, action, next_state)
        terminal = env.terminal(next_state)

        assert next_state.shape == state.shape
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        assert next_state.dtype == jnp.float32
        assert isinstance(obs, jnp.ndarray)
        assert obs.dtype == jnp.float32

        state = next_state


@pytest.mark.parametrize("env_class", [CartPoleFunctional, PendulumFunctional])
def test_vmap(env_class):
    """Tests vmap of functional instance functions with transform."""
    env = env_class()
    num_envs = 10
    rng = jrng.split(jrng.PRNGKey(0), num_envs)

    env.transform(jax.vmap)
    env.transform(jax.jit)
    state = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        action = jnp.array([env.action_space.sample() for _ in range(num_envs)])
        # if isinstance(env.action_space, Discrete):
        #     action = action.reshape((num_envs, 1))
        next_state = env.transition(state, action, None)
        terminal = env.terminal(next_state)
        reward = env.reward(state, action, next_state)

        assert next_state.shape == state.shape
        assert next_state.dtype == jnp.float32
        assert reward.shape == (num_envs,)
        assert reward.dtype == jnp.float32
        assert terminal.shape == (num_envs,)
        assert terminal.dtype == bool
        assert isinstance(obs, jnp.ndarray)
        assert obs.dtype == jnp.float32

        state = next_state
