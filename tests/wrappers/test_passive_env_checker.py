"""Test suite for PassiveEnvChecker wrapper."""

import re
import warnings

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import PassiveEnvChecker
from tests.envs.test_envs import PASSIVE_CHECK_IGNORE_WARNING
from tests.envs.utils import all_testing_initialised_envs
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize(
    "env",
    all_testing_initialised_envs,
    ids=[env.spec.id for env in all_testing_initialised_envs if env.spec is not None],
)
def test_passive_checker_wrapper_warnings(env):
    if env.spec is not None and env.spec.disable_env_checker:
        return

    with warnings.catch_warnings(record=True) as caught_warnings:
        checker_env = PassiveEnvChecker(env)
        checker_env.reset()
        checker_env.step(checker_env.action_space.sample())
        # todo, add check for render, bugged due to mujoco v2/3 and v4 envs

        checker_env.close()

    for warning in caught_warnings:
        if warning.message.args[0] not in PASSIVE_CHECK_IGNORE_WARNING:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")


@pytest.mark.parametrize(
    "env, error_type, message",
    [
        (
            GenericTestEnv(action_space=None),
            AttributeError,
            "The environment must specify an action space. https://gymnasium.farama.org/introduction/create_custom_env/",
        ),
        (
            GenericTestEnv(action_space="error"),
            TypeError,
            "action space does not inherit from `gymnasium.spaces.Space`, actual type: <class 'str'>",
        ),
        (
            GenericTestEnv(observation_space=None),
            AttributeError,
            "The environment must specify an observation space. https://gymnasium.farama.org/introduction/create_custom_env/",
        ),
        (
            GenericTestEnv(observation_space="error"),
            TypeError,
            "observation space does not inherit from `gymnasium.spaces.Space`, actual type: <class 'str'>",
        ),
    ],
)
def test_initialise_failures(env, error_type, message):
    with pytest.raises(error_type, match=f"^{re.escape(message)}$"):
        PassiveEnvChecker(env)

    env.close()


def _reset_failure(self, seed=None, options=None):
    return np.array([-1.0], dtype=np.float32), {}


def _step_failure(self, action):
    return "error"


def test_api_failures():
    env = GenericTestEnv(
        reset_func=_reset_failure,
        step_func=_step_failure,
        metadata={"render_modes": "error"},
    )
    env = PassiveEnvChecker(env)
    assert env.checked_reset is False
    assert env.checked_step is False
    assert env.checked_render is False

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "The obs returned by the `reset()` method is not within the observation space"
        ),
    ):
        env.reset()
    assert env.checked_reset

    with pytest.raises(
        AssertionError,
        match="Expects step result to be a tuple, actual type: <class 'str'>",
    ):
        env.step(env.action_space.sample())
    assert env.checked_step

    with pytest.warns(
        UserWarning,
        match=r"Expects the render_modes to be a sequence \(i\.e\. list, tuple\), actual type: <class 'str'>",
    ):
        env.render()
    assert env.checked_render

    env.close()
