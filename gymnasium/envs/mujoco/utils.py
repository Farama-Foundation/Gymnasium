"""A set of MujocoEnv related utilities, mainly for testing purposes.

Author: @Kallinteris-Andreas
"""

import mujoco
import numpy as np

import gymnasium


def get_state(
    env: gymnasium.envs.mujoco.MujocoEnv,
    state_type: mujoco.mjtState = mujoco.mjtState.mjSTATE_FULLPHYSICS,
):
    """Gets the state of `env`.

    Arguments:
        env: Environment whose state to copy, `env.model` & `env.data` must be accessible.
        state_type: see the [documentation of mjtState](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate) most users can use the default for training purposes or `mujoco.mjtState.mjSTATE_INTEGRATION` for validation purposes.
    """
    assert mujoco.__version__ >= "2.3.6", "Feature requires `mujoco>=2.3.6`"

    state = np.empty(mujoco.mj_stateSize(env.unwrapped.model, state_type))
    mujoco.mj_getState(env.unwrapped.model, env.unwrapped.data, state, state_type)
    return state


def set_state(
    env: gymnasium.envs.mujoco.MujocoEnv,
    state: np.ndarray,
    state_type: mujoco.mjtState = mujoco.mjtState.mjSTATE_FULLPHYSICS,
):
    """Set the state of `env`.

    Arguments:
        env: Environment whose state to set, `env.model` & `env.data` must be accessible.
        state: State to set (generated from get_state).
        state_type: see the [documentation of mjtState](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate) most users can use the default for training purposes or `mujoco.mjtState.mjSTATE_INTEGRATION` for validation purposes.
    """
    assert mujoco.__version__ >= "2.3.6", "Feature requires `mujoco>=2.3.6`"

    mujoco.mj_setState(
        env.unwrapped.model,
        env.unwrapped.data,
        state,
        spec=state_type,
    )
    return state


def check_mujoco_reset_state(
    env: gymnasium.envs.mujoco.MujocoEnv,
    seed=1234,
    state_type: mujoco.mjtState = mujoco.mjtState.mjSTATE_INTEGRATION,
):
    """Asserts that `env.reset()` properly resets the state (not affected by previous steps).

    Note: assuming `check_reset_seed` has passed.

    Arguments:
        env: Environment which is being tested.
        seed: the `seed` used in `env.reset(seed)`.
        state_type: see the [documentation of mjtState](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate).
    """
    env.action_space.seed(seed)
    action = env.action_space.sample()

    env.reset(seed=seed)
    first_reset_state = get_state(env, state_type)
    env.step(action)

    env.reset(seed=seed)
    second_reset_state = get_state(env, state_type)

    assert np.all(first_reset_state == second_reset_state), "reset is not deterministic"
