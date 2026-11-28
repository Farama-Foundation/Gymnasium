import numpy as np
import pytest

import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv, CartPoleVectorEnv

DEFAULT_PARAMS = {"masscart": 1.0, "masspole": 0.1, "length": 0.5}
INITIAL_STATE = np.array([0.0, 0.5, 0.1, -0.3], dtype=np.float64)
PARAM_UPDATES = [
    {"masscart": 2.0},
    {"masspole": 0.2},
    {"length": 1.0},
    {"masscart": 2.0, "masspole": 0.2, "length": 1.0},
]


def _cartpole_euler_step(
    state,
    action,
    *,
    masscart: float,
    masspole: float,
    length: float,
    gravity: float = 9.8,
    force_mag: float = 10.0,
    tau: float = 0.02,
):
    """One Euler cart-pole step with masses derived from the given parameters."""
    x, x_dot, theta, theta_dot = np.asarray(state, dtype=np.float64)
    total_mass = masspole + masscart
    polemass_length = masspole * length
    force = np.sign(np.asarray(action, dtype=np.float64) - 0.5) * force_mag
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + polemass_length * np.square(theta_dot) * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * np.square(costheta) / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
    return np.stack((x, x_dot, theta, theta_dot))


def test_cartpole_default_derived_params():
    """Default total_mass and polemass_length match the default physical parameters."""
    env = CartPoleEnv()
    assert env.total_mass == env.masspole + env.masscart
    assert env.polemass_length == env.masspole * env.length
    assert env.total_mass == 1.1
    assert env.polemass_length == pytest.approx(0.05)
    env.close()


@pytest.mark.parametrize("updates", PARAM_UPDATES)
def test_cartpole_param_assignment_updates_dynamics(updates):
    """Assigning masscart, masspole or length changes the dynamics consistently.

    The constructor does not take these parameters, so the new observation is
    compared with a step computed from the updated masses and length.
    """
    env = gym.make("CartPole-v1", disable_env_checker=True)
    env.reset(seed=0)
    unwrapped = env.unwrapped
    for attr, value in updates.items():
        setattr(unwrapped, attr, value)

    params = {**DEFAULT_PARAMS, **updates}
    assert unwrapped.total_mass == params["masspole"] + params["masscart"]
    assert unwrapped.polemass_length == params["masspole"] * params["length"]

    unwrapped.state = INITIAL_STATE.copy()
    obs, *_ = env.step(1)
    expected = _cartpole_euler_step(INITIAL_STATE, 1, **params)
    np.testing.assert_array_equal(obs, expected.astype(np.float32))
    env.close()


@pytest.mark.parametrize("updates", PARAM_UPDATES)
def test_cartpole_vector_param_assignment_updates_dynamics(updates):
    """The vector CartPole env also derives total_mass and polemass_length."""
    env = CartPoleVectorEnv(num_envs=3)
    env.reset(seed=0)
    for attr, value in updates.items():
        setattr(env, attr, value)

    params = {**DEFAULT_PARAMS, **updates}
    assert env.total_mass == params["masspole"] + params["masscart"]
    assert env.polemass_length == params["masspole"] * params["length"]

    env.state = np.stack([INITIAL_STATE] * env.num_envs, axis=1)
    actions = np.ones(env.num_envs, dtype=np.int64)
    obs, *_ = env.step(actions)
    expected = _cartpole_euler_step(INITIAL_STATE, 1, **params).astype(np.float32)
    for i in range(env.num_envs):
        np.testing.assert_array_equal(obs[i], expected)
    env.close()


def test_cartpole_derived_params_are_read_only():
    """total_mass and polemass_length cannot be assigned independently."""
    env = CartPoleEnv()
    with pytest.raises(AttributeError):
        env.total_mass = 3.0
    with pytest.raises(AttributeError):
        env.polemass_length = 3.0
    env.close()
