"""Test suite for vector NormalizeObservation wrapper."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces, wrappers
from gymnasium.error import InvalidBound
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from tests.testing_env import GenericTestEnv


def create_env():
    return GenericTestEnv(
        observation_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32),
            high=np.array([10, -5, 10], dtype=np.float32),
        )
    )


def test_normalization(
    n_envs: int = 2, convergence_steps: int = 250, testing_steps: int = 100
):
    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    vec_env = wrappers.vector.NormalizeObservation(vec_env)

    vec_env.reset(seed=123)
    vec_env.observation_space.seed(123)
    vec_env.action_space.seed(123)
    for _ in range(convergence_steps):
        vec_env.step(vec_env.action_space.sample())

    observations = []
    for _ in range(testing_steps):
        obs, *_ = vec_env.step(vec_env.action_space.sample())
        observations.append(obs)
    observations = np.array(observations)  # (100, 2, 3)

    mean_obs = np.mean(observations, axis=(0, 1))
    var_obs = np.var(observations, axis=(0, 1))
    assert mean_obs.shape == (3,) and var_obs.shape == (3,)

    assert np.allclose(mean_obs, np.zeros(3), atol=0.15)
    assert np.allclose(var_obs, np.ones(3), atol=0.2)


def test_wrapper_equivalence(
    n_envs: int = 3,
    n_steps: int = 250,
):
    mean_rtol = (np.array([0.1, 0.4, 0.25]),)
    var_rtol = (np.array([0.15, 0.15, 0.18]),)

    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    vec_env = wrappers.vector.NormalizeObservation(vec_env)

    vec_env.reset(seed=123)
    vec_env.observation_space.seed(123)
    vec_env.action_space.seed(123)
    for _ in range(n_steps):
        vec_env.step(vec_env.action_space.sample())

    env = wrappers.Autoreset(create_env())
    env = wrappers.NormalizeObservation(env)
    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(n_steps // n_envs):
        env.step(env.action_space.sample())

    assert np.allclose(env.obs_rms.mean, vec_env.obs_rms.mean, rtol=mean_rtol)
    assert np.allclose(env.obs_rms.var, vec_env.obs_rms.var, rtol=var_rtol)


def test_update_running_mean():
    env = SyncVectorEnv([create_env for _ in range(2)])
    env = wrappers.vector.NormalizeObservation(env)

    # Default value is True
    assert env.update_running_mean

    env.reset()
    for _ in range(100):
        env.step(env.action_space.sample())

    # Disable updating the running mean
    env.update_running_mean = False
    copied_rms_mean = np.copy(env.obs_rms.mean)
    copied_rms_var = np.copy(env.obs_rms.var)

    # Continue stepping through the environment and check that the running mean is not effected
    for _ in range(10):
        env.step(env.action_space.sample())

    assert np.all(copied_rms_mean == env.obs_rms.mean)
    assert np.all(copied_rms_var == env.obs_rms.var)

    # Re-enable updating the running mean
    env.update_running_mean = True

    for _ in range(10):
        env.step(env.action_space.sample())

    assert np.any(copied_rms_mean != env.obs_rms.mean)
    assert np.any(copied_rms_var != env.obs_rms.var)


def test_observation_space_and_dtype():
    vec_env = SyncVectorEnv([create_env for _ in range(2)])
    vec_env = wrappers.vector.NormalizeObservation(vec_env)

    assert vec_env.single_observation_space.dtype == np.float32
    assert np.all(vec_env.single_observation_space.low == -np.inf)
    assert np.all(vec_env.single_observation_space.high == np.inf)

    obs, _ = vec_env.reset(seed=123)
    assert obs.dtype == np.float32

    obs, *_ = vec_env.step(vec_env.action_space.sample())
    assert obs.dtype == np.float32


@pytest.mark.parametrize("epsilon", [0.0, -1e-8, -1.0])
def test_non_positive_epsilon_is_rejected(epsilon):
    """Matches the same check on the non-vector wrapper, which shares this scaling."""
    vec_env = SyncVectorEnv([create_env])
    with pytest.raises(InvalidBound, match="`epsilon` should be strictly positive"):
        wrappers.vector.NormalizeObservation(vec_env, epsilon=epsilon)
    vec_env.close()


@pytest.mark.parametrize(
    "autoreset_mode",
    [
        AutoresetMode.NEXT_STEP,
        AutoresetMode.SAME_STEP,
        pytest.param(
            AutoresetMode.DISABLED,
            marks=pytest.mark.xfail(
                strict=True,
                reason="`NormalizeObservation` rejects `AutoresetMode.DISABLED` even though `reset` supports a full `reset_mask`",
            ),
        ),
    ],
)
def test_equivalence_with_wrapper_autoreset_modes(
    autoreset_mode: AutoresetMode,
    env_id: str = "CartPole-v1",
    num_steps: int = 50,
    max_episode_steps: int = 7,
):
    """With a single sub-environment, the vector wrapper should exactly match `NormalizeObservation` within the vector env."""
    if autoreset_mode == AutoresetMode.SAME_STEP:
        # `info["final_obs"]` isn't normalized, therefore, same-step autoreset is rejected
        with pytest.raises(ValueError, match="Expected autoreset_mode to be"):
            wrappers.vector.NormalizeObservation(
                gym.make_vec(
                    env_id,
                    num_envs=1,
                    vectorization_mode="sync",
                    vector_kwargs={"autoreset_mode": autoreset_mode},
                )
            )
        return

    vec_env = wrappers.vector.NormalizeObservation(
        gym.make_vec(
            env_id,
            num_envs=1,
            vectorization_mode="sync",
            vector_kwargs={"autoreset_mode": autoreset_mode},
            max_episode_steps=max_episode_steps,
        )
    )
    per_env = gym.make_vec(
        env_id,
        num_envs=1,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": autoreset_mode},
        wrappers=(wrappers.NormalizeObservation,),
        max_episode_steps=max_episode_steps,
    )

    vec_obs, _ = vec_env.reset(seed=123)
    per_env_obs, _ = per_env.reset(seed=123)
    assert data_equivalence(vec_obs, per_env_obs)

    vec_env.action_space.seed(123)
    num_episode_ends = 0
    for _ in range(num_steps):
        action = vec_env.action_space.sample()
        vec_obs, vec_rew, vec_term, vec_trunc, _ = vec_env.step(action)
        per_env_obs, per_env_rew, per_env_term, per_env_trunc, _ = per_env.step(action)

        assert data_equivalence(vec_obs, per_env_obs)
        assert data_equivalence(vec_rew, per_env_rew)
        assert data_equivalence(vec_term, per_env_term)
        assert data_equivalence(vec_trunc, per_env_trunc)

        dones = np.logical_or(vec_term, vec_trunc)
        num_episode_ends += int(np.sum(dones))
        if autoreset_mode == AutoresetMode.DISABLED and np.any(dones):
            vec_obs, _ = vec_env.reset(options={"reset_mask": dones})
            per_env_obs, _ = per_env.reset(options={"reset_mask": dones})
            assert data_equivalence(vec_obs, per_env_obs)

    assert num_episode_ends >= num_steps // (max_episode_steps + 1)
    assert np.allclose(vec_env.obs_rms.mean, per_env.envs[0].obs_rms.mean)
    assert np.allclose(vec_env.obs_rms.var, per_env.envs[0].obs_rms.var)
    assert vec_env.obs_rms.count == per_env.envs[0].obs_rms.count

    vec_env.close()
    per_env.close()


@pytest.mark.xfail(
    strict=True,
    reason="`NormalizeObservation` rejects `AutoresetMode.DISABLED` even though `reset` supports a full `reset_mask`",
)
def test_disabled_autoreset_rejects_partial_reset(n_envs: int = 2):
    vec_env = wrappers.vector.NormalizeObservation(
        SyncVectorEnv(
            [create_env for _ in range(n_envs)], autoreset_mode=AutoresetMode.DISABLED
        )
    )
    vec_env.reset(seed=123)
    with pytest.raises(ValueError, match="does not support partial resets"):
        vec_env.reset(options={"reset_mask": np.array([True, False])})

    # A full reset through `reset_mask` is still allowed
    vec_env.reset(options={"reset_mask": np.array([True, True])})
    vec_env.close()
