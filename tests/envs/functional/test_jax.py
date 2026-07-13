import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import jax.random as jrng  # noqa: E402
import numpy as np  # noqa: E402

from gymnasium.envs.phys2d.cartpole import (  # noqa: E402
    CartPoleFunctional,
    CartPoleJaxVectorEnv,
)
from gymnasium.envs.phys2d.pendulum import (  # noqa: E402
    PendulumFunctional,
    PendulumJaxVectorEnv,
)


@pytest.mark.parametrize("env_class", [CartPoleFunctional, PendulumFunctional])
def test_normal(env_class):
    env = env_class()
    rng = jrng.PRNGKey(0)

    state = env.initial(rng)
    env.action_space.seed(0)

    for _ in range(10):
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
    env = env_class()
    rng = jrng.PRNGKey(0)

    env.transform(jax.jit)
    state = env.initial(rng)
    env.action_space.seed(0)

    for _ in range(10):
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
    env = env_class()
    num_envs = 10
    rng = jrng.split(jrng.PRNGKey(0), num_envs)

    env.transform(jax.vmap)
    env.transform(jax.jit)
    state = env.initial(rng)
    env.action_space.seed(0)

    for _ in range(10):
        obs = env.observation(state, None)
        action = jnp.array([env.action_space.sample() for _ in range(num_envs)])
        # if isinstance(env.action_space, Discrete):
        #     action = action.reshape((num_envs, 1))
        next_state = env.transition(state, action, None)
        terminal = env.terminal(next_state, None)
        reward = env.reward(state, action, next_state, None)

        assert next_state.shape == state.shape
        assert next_state.dtype == jnp.float32
        assert reward.shape == (num_envs,)
        assert reward.dtype == jnp.float32
        assert terminal.shape == (num_envs,)
        assert terminal.dtype == np.bool_
        assert isinstance(obs, jax.Array)
        assert obs.dtype == jnp.float32

        state = next_state


@pytest.mark.parametrize("env_class", [CartPoleJaxVectorEnv, PendulumJaxVectorEnv])
def test_vectorized(env_class):
    env = env_class(num_envs=10)
    env.action_space.seed(0)

    obs, info = env.reset(seed=0)
    assert obs.shape == (10,) + env.single_observation_space.shape
    assert isinstance(obs, jax.Array)
    assert isinstance(info, dict)

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (10,) + env.single_observation_space.shape
        assert isinstance(obs, jax.Array)
        assert reward.shape == (10,)
        assert isinstance(reward, jax.Array)
        assert terminated.shape == (10,)
        assert isinstance(terminated, jax.Array)
        assert truncated.shape == (10,)
        assert isinstance(truncated, jax.Array)
        assert isinstance(info, dict)

        # These were removed in the new autoreset order
        assert "final_observation" not in info
        assert "final_info" not in info
        assert "_final_observation" not in info
        assert "_final_info" not in info


@pytest.mark.parametrize(
    "env_class, options",
    [
        (CartPoleJaxVectorEnv, {"gravity": 15.0}),
        (PendulumJaxVectorEnv, {}),  # Pendulum has no gravity param
    ],
)
def test_jax_vector_env_reset_accepts_options(env_class, options):
    env = env_class(num_envs=4)

    # Should not raise
    obs, info = env.reset(seed=0, options=options)

    assert isinstance(obs, jax.Array)
    assert isinstance(info, dict)


@pytest.mark.parametrize("env_class", [CartPoleJaxVectorEnv, PendulumJaxVectorEnv])
def test_jax_vector_env_reset_without_options_is_stable(env_class):
    env = env_class(num_envs=4)

    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)

    assert jnp.allclose(obs1, obs2)


def test_jax_vector_env_reset_options_change_params():
    env = CartPoleJaxVectorEnv(num_envs=4)

    obs1, _ = env.reset(seed=123, options={"x_init": 0.05})
    obs2, _ = env.reset(seed=123, options={"x_init": 0.5})

    assert not jnp.allclose(obs1, obs2)


def test_jax_vector_env_reset_uses_generate_params(monkeypatch):
    env = CartPoleJaxVectorEnv(num_envs=4)

    called = {"count": 0}

    def _generate_params(**kwargs):
        called["count"] += 1
        return env.func_env.get_default_params(**kwargs)

    monkeypatch.setattr(env.func_env, "generate_params", _generate_params)

    env.reset(seed=0, options={"gravity": 15.0})

    assert called["count"] == 1
