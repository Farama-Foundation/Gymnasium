"""Test the functional jax environment."""

import numpy as np
import pytest


pytest.skip(
    "Github CI is running forever for the tests in this file.", allow_module_level=True
)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import jax.random as jrng  # noqa: E402

import gymnasium as gym  # noqa: E402
from gymnasium.envs.phys2d.cartpole import CartPoleFunctional  # noqa: E402
from gymnasium.envs.phys2d.pendulum import PendulumFunctional  # noqa: E402


@pytest.mark.parametrize("env_class", [CartPoleFunctional, PendulumFunctional])
def test_without_transform(env_class):
    """Tests the environment without transforming the environment."""
    env = env_class()
    rng = jrng.PRNGKey(0)

    state = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state, rng)
        action = env.action_space.sample()
        next_state = env.transition(state, action, rng)
        reward = env.reward(state, action, next_state, rng)
        terminal = env.terminal(next_state, rng)

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
        assert isinstance(obs, jax.Array)
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
        obs = env.observation(state, rng)
        action = env.action_space.sample()
        next_state = env.transition(state, action, rng)
        reward = env.reward(state, action, next_state, rng)
        terminal = env.terminal(next_state, rng)

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
        assert isinstance(obs, jax.Array)
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
        obs = env.observation(state, rng)
        action = jnp.array([env.action_space.sample() for _ in range(num_envs)])
        # if isinstance(env.action_space, Discrete):
        #     action = action.reshape((num_envs, 1))
        next_state = env.transition(state, action, rng)
        terminal = env.terminal(next_state, rng)
        reward = env.reward(state, action, next_state, rng)

        assert next_state.shape == state.shape
        assert next_state.dtype == jnp.float32
        assert reward.shape == (num_envs,)
        assert reward.dtype == jnp.float32
        assert terminal.shape == (num_envs,)
        assert terminal.dtype == bool
        assert isinstance(obs, jax.Array)
        assert obs.dtype == jnp.float32

        state = next_state


@pytest.mark.parametrize("vectorization_mode", ["vector_entry_point", "sync", "async"])
def test_equal_episode_length(vectorization_mode: str):
    """Tests that the number of steps in an episode is the same."""

    env = gym.make_vec("phys2d/Pendulum-v0", 2, vectorization_mode=vectorization_mode)
    # By default, the total number of steps per episode is 200

    expected_dones = [199, 399, 599, 799, 999]

    env.action_space.seed(0)

    env.reset()

    for t in range(1000):

        actions = env.action_space.sample()

        next_obs, reward, term, trunc, info = env.step(actions)

        done = np.logical_or(term, trunc).any()

        if done:
            assert t in expected_dones
        else:
            assert t not in expected_dones

        if done:
            obs, *_ = env.step(actions)
