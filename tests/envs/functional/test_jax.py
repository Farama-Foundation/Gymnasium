import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import jax.random as jrng  # noqa: E402
import numpy as np  # noqa: E402

from gymnasium.envs.phys2d.cartpole import (  # noqa: E402
    CartPoleFunctional,
    CartPoleJaxEnv,
    CartPoleJaxVectorEnv,
)
from gymnasium.envs.phys2d.pendulum import (  # noqa: E402
    PendulumFunctional,
    PendulumJaxEnv,
    PendulumJaxVectorEnv,
)
from gymnasium.envs.tabular.blackjack import BlackJackJaxEnv  # noqa: E402


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


def test_cartpole_matches_numpy_cartpole():
    """Checks that the jax CartPole has the same dynamics as the numpy CartPole."""
    from gymnasium.envs.classic_control.cartpole import CartPoleEnv

    func_env = CartPoleFunctional()
    numpy_env = CartPoleEnv()
    numpy_env.reset(seed=0)

    state = jnp.array([0.01, -0.02, 0.03, 0.01], dtype=jnp.float32)
    numpy_env.state = np.array(state, dtype=np.float64)
    for action in [1, 0, 1, 1, 0]:
        state = func_env.transition(state, action)
        numpy_obs, *_ = numpy_env.step(action)

        assert np.allclose(state, numpy_obs, atol=1e-5)


@pytest.mark.parametrize(
    "env_class, kwargs",
    [
        (CartPoleJaxEnv, {"sutton_barto_reward": True, "masspole": 0.2}),
        (PendulumJaxEnv, {"g": 0.0}),
        (BlackJackJaxEnv, {"natural": True, "sutton_and_barto": False}),
        (CartPoleJaxVectorEnv, {"num_envs": 2, "sutton_barto_reward": True}),
        (PendulumJaxVectorEnv, {"num_envs": 2, "g": 0.0}),
    ],
)
def test_env_kwargs(env_class, kwargs):
    """Checks that the environment kwargs are used to set the environment parameters."""
    env = env_class(**kwargs)

    params = env.func_env.get_default_params(
        **{key: value for key, value in kwargs.items() if key != "num_envs"}
    )
    for key, value in kwargs.items():
        if key != "num_envs":
            assert getattr(params, key) == value

    env.reset(seed=0)
    env.step(env.action_space.sample())


def test_cartpole_derived_params():
    params = CartPoleFunctional().get_default_params(masspole=0.2, length=1.0)
    assert np.isclose(params.total_mass, 1.2)
    assert np.isclose(params.polemass_length, 0.2)


def test_cartpole_sutton_barto_reward_kwarg():
    env = CartPoleJaxEnv(sutton_barto_reward=True)
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(1)
    assert not terminated and reward == 0.0
