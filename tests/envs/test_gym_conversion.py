import pytest

import gymnasium
from gymnasium.utils.env_checker import check_env

pytest.importorskip("gym")

import gym  # noqa: E402, isort: skip

# We do not test Atari environment's here because we check all variants of Pong in test_envs.py (There are too many Atari environments)
ALL_GYM_ENVS = [
    env_id
    for env_id, spec in gym.envs.registry.items()
    if "ale_py" not in spec.entry_point
]


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
