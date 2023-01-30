import numpy as np
import pytest

import gymnasium as gym
from gymnasium import envs
from gymnasium.envs.registration import get_env_id, parse_env_id
from tests.envs.utils import mujoco_testing_env_ids


EPS = 1e-6


def verify_environments_match(
    old_env_id: str, new_env_id: str, seed: int = 1, num_actions: int = 1000
):
    """Verifies with two environment ids (old and new) are identical in obs, reward and done
    (except info where all old info must be contained in new info)."""
    old_env = envs.make(old_env_id, disable_env_checker=True)
    new_env = envs.make(new_env_id, disable_env_checker=True)

    old_reset_obs, old_info = old_env.reset(seed=seed)
    new_reset_obs, new_info = new_env.reset(seed=seed)

    np.testing.assert_allclose(old_reset_obs, new_reset_obs)

    for i in range(num_actions):
        action = old_env.action_space.sample()
        old_obs, old_reward, old_terminated, old_truncated, old_info = old_env.step(
            action
        )
        new_obs, new_reward, new_terminated, new_truncated, new_info = new_env.step(
            action
        )

        np.testing.assert_allclose(old_obs, new_obs, atol=EPS)
        np.testing.assert_allclose(old_reward, new_reward, atol=EPS)
        np.testing.assert_equal(old_terminated, new_terminated)
        np.testing.assert_equal(old_truncated, new_truncated)

        for key in old_info:
            np.testing.assert_allclose(old_info[key], new_info[key], atol=EPS)

        if old_terminated or old_truncated:
            break


EXCLUDE_POS_FROM_OBS = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "Humanoid",
    "Swimmer",
    "Walker2d",
]


@pytest.mark.parametrize(
    "env_id",
    mujoco_testing_env_ids,
    ids=mujoco_testing_env_ids,
)
def test_obs_space_mujoco_environments(env_id: str):
    """Check that the returned observations are contained in the observation space of the environment"""
    env = gym.make(env_id, disable_env_checker=True)
    reset_obs, info = env.reset()
    assert env.observation_space.contains(
        reset_obs
    ), f"Observation returned by reset() of {env_id} is not contained in the default observation space {env.observation_space}."

    action = env.action_space.sample()
    step_obs, _, _, _, _ = env.step(action)
    assert env.observation_space.contains(
        step_obs
    ), f"Observation returned by step(action) of {env_id} is not contained in the default observation space {env.observation_space}."

    ns, name, version = parse_env_id(env_id)
    if name in EXCLUDE_POS_FROM_OBS and (version == 4 or version == 3):
        env = gym.make(
            env_id,
            disable_env_checker=True,
            exclude_current_positions_from_observation=False,
        )
        reset_obs, info = env.reset()
        assert env.observation_space.contains(
            reset_obs
        ), f"Observation of {env_id} is not contained in the default observation space {env.observation_space} when excluding current position from observation."

        step_obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(
            step_obs
        ), f"Observation returned by step(action) of {env_id} is not contained in the default observation space {env.observation_space} when excluding current position from observation."

    # Ant-v4 has the option of including contact forces in the observation space with the use_contact_forces argument
    if name == "Ant" and version == 4:
        env = gym.make(env_id, disable_env_checker=True, use_contact_forces=True)
        reset_obs, info = env.reset()
        assert env.observation_space.contains(
            reset_obs
        ), f"Observation of {env_id} is not contained in the default observation space {env.observation_space} when using contact forces."

        step_obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(
            step_obs
        ), f"Observation returned by step(action) of {env_id} is not contained in the default observation space {env.observation_space} when using contact forces."


MUJOCO_V2_V3_ENV_IDS = []
for _env_id in mujoco_testing_env_ids:
    _, name, version = parse_env_id(_env_id)

    if version == 2 and get_env_id(None, name, 3) in gym.envs.registry:
        MUJOCO_V2_V3_ENV_IDS.append(_env_id)


@pytest.mark.parametrize("env_id", MUJOCO_V2_V3_ENV_IDS)
def test_mujoco_v2_to_v3_conversion(env_id: str):
    """Checks that all v2 mujoco environments are the same as v3 environments."""
    verify_environments_match(f"{env_id}-v2", f"{env_id}-v3")


@pytest.mark.parametrize("env_id", MUJOCO_V2_V3_ENV_IDS)
def test_mujoco_incompatible_v3_to_v2(env_id: str):
    """Checks that the v3 environment are slightly different from v2, (v3 has additional info keys that v2 does not)."""
    with pytest.raises(KeyError):
        verify_environments_match(f"{env_id}-v3", f"{env_id}-v2")
