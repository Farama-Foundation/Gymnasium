import warnings

import pytest

import gymnasium
from gymnasium.utils.env_checker import check_env
from tests.envs.test_envs import CHECK_ENV_IGNORE_WARNINGS

pytest.importorskip("gym")

import gym  # noqa: E402, isort: skip

# We do not test Atari environment's here because we check all variants of Pong in test_envs.py (There are too many Atari environments)
ALL_GYM_ENVS = [
    env_id
    for env_id, spec in gym.envs.registry.items()
    if ("ale_py" not in spec.entry_point or "Pong" in env_id)
]


@pytest.mark.parametrize(
    "env_id", ALL_GYM_ENVS, ids=[env_id for env_id in ALL_GYM_ENVS]
)
def test_gym_conversion_by_id(env_id):
    env = gymnasium.make("GymV26Environment-v0", env_id=env_id).unwrapped
    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env)
    for warning in caught_warnings:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise gymnasium.error.Error(f"Unexpected warning: {warning.message}")


@pytest.mark.parametrize(
    "env_id", ALL_GYM_ENVS, ids=[env_id for env_id in ALL_GYM_ENVS]
)
def test_gym_conversion_instantiated(env_id):
    env = gym.make(env_id)
    env = gymnasium.make("GymV26Environment-v0", env=env).unwrapped
    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env)
    for warning in caught_warnings:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise gymnasium.error.Error(f"Unexpected warning: {warning.message}")
