import gym
import pytest

import gymnasium
from gymnasium.utils.env_checker import check_env

pytest.importorskip("gym")
ALL_GYM_ENVS = gym.envs.registry.keys()


@pytest.mark.parametrize(
    "env_id", ALL_GYM_ENVS, ids=[env_id for env_id in ALL_GYM_ENVS]
)
def test_gym_conversion_by_id(env_id):
    env = gymnasium.make("GymV26Environment-v0", env_id=env_id)
    check_env(env)


@pytest.mark.parametrize(
    "env_id", ALL_GYM_ENVS, ids=[env_id for env_id in ALL_GYM_ENVS]
)
def test_gym_conversion_instantiated(env_id):
    env = gym.make(env_id)
    env = gymnasium.make("GymV26Environment-v0", env=env)
    check_env(env)
