"""Tests parameter forwarding in the functional JAX adapters."""

from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from gymnasium.envs.functional_jax_env import (  # noqa: E402
    FunctionalJaxEnv,
    FunctionalJaxVectorEnv,
)
from gymnasium.experimental.functional import FuncEnv  # noqa: E402


class ParameterizedFuncEnv(FuncEnv):
    """Small functional environment that records rendering parameters."""

    observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self, params: dict[str, float]):
        super().__init__(params=params)
        self.render_params = None
        self.close_params = None

    def initial(self, rng: Any, params=None):
        return jnp.array([params["value"]], dtype=jnp.float32)

    def transition(self, state, action, rng: Any, params=None):
        return state + action * params["value"]

    def observation(self, state, rng: Any, params=None):
        return state + params["value"]

    def reward(self, state, action, next_state, rng: Any, params=None):
        return jnp.asarray(params["value"], dtype=jnp.float32)

    def terminal(self, state, rng: Any, params=None):
        return jnp.asarray(False)

    def state_info(self, state, params=None):
        return {"param": jnp.asarray(params["value"])}

    def transition_info(self, state, action, next_state, params=None):
        return {"param": jnp.asarray(params["value"])}

    def render_init(self, params=None):
        return object()

    def render_image(self, state, render_state, params=None):
        self.render_params = params
        return render_state, np.zeros((1, 1, 3), dtype=np.uint8)

    def render_close(self, render_state, params=None):
        self.close_params = params


def test_functional_jax_env_forwards_params():
    params = {"value": 2.0}
    func_env = ParameterizedFuncEnv(params)
    env = FunctionalJaxEnv(func_env, render_mode="rgb_array")

    observation, info = env.reset(seed=0)
    assert observation.shape == (1,)
    assert info["param"] == params["value"]

    _, reward, terminated, truncated, info = env.step(jnp.array([0.5]))
    assert reward == params["value"]
    assert not terminated
    assert not truncated
    assert info["param"] == params["value"]

    env.render()
    env.close()
    assert func_env.render_params is params
    assert func_env.close_params is params


def test_functional_jax_vector_env_broadcasts_params():
    params = {"value": 2.0}
    func_env = ParameterizedFuncEnv(params)
    env = FunctionalJaxVectorEnv(func_env, num_envs=2, render_mode="rgb_array")

    observation, info = env.reset(seed=0)
    assert observation.shape == (2, 1)
    np.testing.assert_array_equal(info["param"], [2.0, 2.0])

    _, reward, terminated, truncated, info = env.step(jnp.array([[0.5], [0.5]]))
    np.testing.assert_array_equal(reward, [2.0, 2.0])
    assert not terminated.any()
    assert not truncated.any()
    np.testing.assert_array_equal(info["param"], [2.0, 2.0])

    env.render()
    env.close()
    assert func_env.render_params is params
    assert func_env.close_params is params


@pytest.mark.parametrize("env_id", ["tabular/Blackjack-v0", "tabular/CliffWalking-v0"])
def test_registered_functional_jax_env(env_id):
    env = gym.make(env_id)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()
