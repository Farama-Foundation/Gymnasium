import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
import pytest

from gymnasium.envs.jax_toy_text.blackjack import BlackJackF  # noqa: E402



def test_normal_BlackJackF():
    env = BlackJackF()
    rng = jrng.PRNGKey(0)

    state, rng = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        action = env.action_space.sample()
        next_state, rng = env.transition(state, action, rng)
        reward = env.reward(state, action, next_state)
        terminal = env.terminal(next_state)

        assert len(state) == len(next_state)
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        print(next_state)
       
        assert next_state[0].dtype == jnp.float32
        assert next_state[1].dtype == jnp.float32
        assert next_state[2].dtype == jnp.int32
        assert next_state[3].dtype == jnp.int32
        assert next_state[4].dtype == jnp.int32


        assert rng.dtype == jnp.uint32
        assert obs[0].dtype == jnp.float32
        assert obs[1].dtype == jnp.float32
        assert obs[2].dtype == jnp.float32

        state = next_state


def test_jit_BlackJackF():
    env = BlackJackF()
    rng = jrng.PRNGKey(0)

    env.transform(jax.jit)
    state,rng  = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        action = env.action_space.sample()
        next_state, rng = env.transition(state, action, rng)
        reward = env.reward(state, action, next_state)
        terminal = env.terminal(next_state)

        assert len(state) == len(next_state)
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        assert next_state[0].dtype == jnp.float32
        assert next_state[1].dtype == jnp.float32
        assert next_state[2].dtype == jnp.int32
        assert next_state[3].dtype == jnp.int32
        assert next_state[4].dtype == jnp.int32

        assert rng.dtype == jnp.uint32
        assert obs[0].dtype == jnp.float32
        assert obs[1].dtype == jnp.float32
        assert obs[2].dtype == jnp.float32

        state = next_state






def test_vmap_BlackJack_():
    env = BlackJackF()
    num_envs = 10
    rng = jrng.split(jrng.PRNGKey(0), num_envs)

    env.transform(jax.vmap)
    env.transform(jax.jit)
    state, rng = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        action = jnp.array([env.action_space.sample() for _ in range(num_envs)])
        # if isinstance(env.action_space, Discrete):
        #     action = action.reshape((num_envs, 1))
        next_state, rng = env.transition(state, action, rng)
        terminal = env.terminal(next_state)
        reward = env.reward(state, action, next_state)

        assert len(next_state) == len(state)
        #assert next_state.dtype == jnp.float32
        assert reward.shape == (num_envs,)
        assert reward.dtype == jnp.float32
        assert terminal.shape == (num_envs,)
        assert terminal.dtype == bool
        assert isinstance(obs, tuple)
        assert obs[0].dtype == jnp.float32
        assert obs[1].dtype == jnp.float32
        assert obs[2].dtype == jnp.float32

        state = next_state
