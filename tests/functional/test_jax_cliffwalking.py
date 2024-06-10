"""Tests for Jax cliffwalking functional env."""

import pytest


jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import jax.random as jrng  # noqa: E402

from gymnasium.envs.tabular.cliffwalking import CliffWalkingFunctional  # noqa: E402


def test_normal_CliffWalkingFunctional():
    """Tests to ensure that cliffwalking env step and reset functions return the correct types."""
    env = CliffWalkingFunctional()
    rng = jrng.PRNGKey(0)

    split_rng, rng = jrng.split(rng)

    state = env.initial(split_rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state, rng)
        action = env.action_space.sample()

        split_rng, rng = jrng.split(rng)

        next_state = env.transition(state, action, split_rng)
        reward = env.reward(state, action, next_state, split_rng)
        terminal = env.terminal(next_state, split_rng)

        assert len(state) == len(next_state)
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        assert next_state[0].dtype == jnp.int32
        assert next_state[1].dtype == jnp.int32
        assert next_state[2].dtype == bool

        assert rng.dtype == jnp.uint32
        assert obs.dtype == jnp.int32

        state = next_state


def test_jit_CliffWalkingFunctional():
    """Tests the Jax CliffWalkingFunctional env, but in a jitted context."""
    env = CliffWalkingFunctional()
    rng = jrng.PRNGKey(0)
    env.transform(jax.jit)

    split_rng, rng = jrng.split(rng)

    state = env.initial(split_rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state, rng)
        action = env.action_space.sample()
        split_rng, rng = jrng.split(rng)
        next_state = env.transition(state, action, split_rng)
        reward = env.reward(state, action, next_state, split_rng)
        terminal = env.terminal(next_state, split_rng)

        assert len(state) == len(next_state)
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        assert next_state[0].dtype == jnp.int32
        assert next_state[1].dtype == jnp.int32
        assert next_state[2].dtype == bool

        assert rng.dtype == jnp.uint32
        assert obs.dtype == jnp.int32

        state = next_state


def test_vmap_BlackJack():
    """Tests the Jax CliffWalking env with vmap."""
    env = CliffWalkingFunctional()
    num_envs = 10
    rng, *split_rng = jrng.split(
        jrng.PRNGKey(0), num_envs + 1
    )  # this plus 1 is important because we want
    # num_envs subkeys and a main entropy source key which necessitates an additional key

    env.transform(jax.vmap)
    env.transform(jax.jit)
    state = env.initial(jnp.array(split_rng))
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state, jnp.array(split_rng))
        action = jnp.array([env.action_space.sample() for _ in range(num_envs)])
        # if isinstance(env.action_space, Discrete):
        #     action = action.reshape((num_envs, 1))
        rng, *split_rng = jrng.split(rng, num_envs + 1)
        next_state = env.transition(state, action, jnp.array(split_rng))
        terminal = env.terminal(next_state, jnp.array(split_rng))
        reward = env.reward(state, action, next_state, jnp.array(split_rng))

        assert len(next_state) == len(state)
        assert reward.shape == (num_envs,)
        assert reward.dtype == jnp.float32
        assert terminal.shape == (num_envs,)
        assert terminal.dtype == bool
        assert isinstance(obs, jax.Array)
        assert obs[0].dtype == jnp.int32
        assert obs[1].dtype == jnp.int32
        assert obs[2].dtype == jnp.int32

        state = next_state
