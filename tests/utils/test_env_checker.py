"""Tests that the `env_checker` runs as expects and all errors are possible."""

import re
import warnings
from collections.abc import Callable

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.utils.env_checker import (
    check_env,
    check_reset_options,
    check_reset_return_info_deprecation,
    check_reset_return_type,
    check_reset_seed_determinism,
    check_seed_deprecation,
    check_step_determinism,
)
from tests.testing_env import GenericTestEnv


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]


def _no_error_warnings_envs():
    yield gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    yield gym.make("MountainCar-v0", disable_env_checker=True).unwrapped
    space_a = spaces.Discrete(10)
    space_b = spaces.Box(np.zeros(2, np.float32), np.ones(2, np.float32))
    yield GenericTestEnv(observation_space=spaces.Dict(a=space_a, b=space_b))
    yield GenericTestEnv(observation_space=spaces.Tuple([space_a, space_b]))
    yield GenericTestEnv(
        observation_space=spaces.Dict(a=spaces.Tuple([space_a, space_b]), b=space_b)
    )


@pytest.mark.parametrize("env", _no_error_warnings_envs())
def test_no_error_warnings(env):
    """A full version of this test with all gymnasium envs is run in tests/envs/test_envs.py."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env)
    caught_warnings = [
        warning
        for warning in caught_warnings
        if str(warning.message) not in CHECK_ENV_IGNORE_WARNINGS
    ]

    assert len(caught_warnings) == 0, [warning.message for warning in caught_warnings]


def _no_super_reset(self, seed=None, options=None):
    self.np_random.random()  # generates a new prng
    # generate seed deterministic result
    self.observation_space.seed(0)
    return self.observation_space.sample(), {}


def _super_reset_fixed(self, seed=None, options=None):
    # Call super that ignores the seed passed, use fixed seed
    super(GenericTestEnv, self).reset(seed=1)
    # deterministic output
    self.observation_space._np_random = self.np_random
    return self.observation_space.sample(), {}


def _reset_default_seed(self: GenericTestEnv, seed=23, options=None):
    super(GenericTestEnv, self).reset(seed=seed)
    self.observation_space._np_random = (  # pyright: ignore [reportPrivateUsage]
        self.np_random
    )
    return self.observation_space.sample(), {}


@pytest.mark.parametrize(
    "test,func,message",
    [
        [
            gym.error.Error,
            lambda self: (self.observation_space.sample(), {}),
            "The `reset` method does not provide a `seed` or `**kwargs` keyword argument.",
        ],
        [
            AssertionError,
            lambda self, seed, *_: (self.observation_space.sample(), {}),
            "Expects the random number generator to have been generated given a seed was passed to reset. Most likely the environment reset function does not call `super().reset(seed=seed)`.",
        ],
        [
            AssertionError,
            _no_super_reset,
            "Most likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`.",
        ],
        [
            AssertionError,
            _super_reset_fixed,
            "Most likely the environment reset function does not call `super().reset(seed=seed)` as the random number generators are not different when different seeds are passed to `env.reset`.",
        ],
        [
            UserWarning,
            _reset_default_seed,
            "The default seed argument in reset should be `None`, otherwise the environment will by default always be deterministic. Actual default: 23",
        ],
    ],
)
def test_check_reset_seed_determinism(test, func: Callable, message: str):
    """Tests the check reset seed function works as expected."""
    if test is UserWarning:
        with pytest.warns(
            UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
        ):
            check_reset_seed_determinism(GenericTestEnv(reset_func=func))
    else:
        with pytest.raises(test, match=f"^{re.escape(message)}$"):
            check_reset_seed_determinism(GenericTestEnv(reset_func=func))


def _deprecated_return_info(
    self, return_info: bool = False
) -> tuple[ObsType, dict] | ObsType:
    """function to simulate the signature and behavior of a `reset` function with the deprecated `return_info` optional argument"""
    if return_info:
        return self.observation_space.sample(), {}
    else:
        return self.observation_space.sample()


def _reset_var_keyword_kwargs(self, kwargs):
    return self.observation_space.sample(), {}


def _reset_return_info_type(self, seed=None, options=None):
    """Returns a `list` instead of a `tuple`. This function is used to make sure `env_checker` correctly
    checks that the return type of `env.reset()` is a `tuple`"""
    return [self.observation_space.sample(), {}]


def _reset_return_info_length(self, seed=None, options=None):
    return 1, 2, 3


def _return_info_obs_outside(self, seed=None, options=None):
    return self.observation_space.sample() + self.observation_space.high, {}


def _return_info_not_dict(self, seed=None, options=None):
    return self.observation_space.sample(), ["key", "value"]


@pytest.mark.parametrize(
    "test,func,message",
    [
        [
            AssertionError,
            _reset_return_info_type,
            "The result returned by `env.reset()` was not a tuple of the form `(obs, info)`, where `obs` is a observation and `info` is a dictionary containing additional information. Actual type: `<class 'list'>`",
        ],
        [
            AssertionError,
            _reset_return_info_length,
            "Calling the reset method did not return a 2-tuple, actual length: 3",
        ],
        [
            AssertionError,
            _return_info_obs_outside,
            "The first element returned by `env.reset()` is not within the observation space.",
        ],
        [
            AssertionError,
            _return_info_not_dict,
            "The second element returned by `env.reset()` was not a dictionary, actual type: <class 'list'>",
        ],
    ],
)
def test_check_reset_return_type(test, func: Callable, message: str):
    """Tests the check `env.reset()` function has a correct return type."""

    with pytest.raises(test, match=f"^{re.escape(message)}$"):
        check_reset_return_type(GenericTestEnv(reset_func=func))


@pytest.mark.parametrize(
    "test,func,message",
    [
        [
            UserWarning,
            _deprecated_return_info,
            "`return_info` is deprecated as an optional argument to `reset`. `reset`"
            "should now always return `obs, info` where `obs` is an observation, and `info` is a dictionary"
            "containing additional information.",
        ],
    ],
)
def test_check_reset_return_info_deprecation(test, func: Callable, message: str):
    """Tests that return_info has been correct deprecated as an argument to `env.reset()`."""

    with pytest.warns(test, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"):
        check_reset_return_info_deprecation(GenericTestEnv(reset_func=func))


def test_check_seed_deprecation():
    """Tests that `check_seed_deprecation()` throws a warning if `env.seed()` has not been removed."""

    message = """Official support for the `seed` function is dropped. Standard practice is to reset gymnasium environments using `env.reset(seed=<desired seed>)`"""

    env = GenericTestEnv()

    def seed(seed):
        return

    with pytest.warns(
        UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
    ):
        env.seed = seed
        assert callable(env.seed)
        check_seed_deprecation(env)

    with warnings.catch_warnings(record=True) as caught_warnings:
        env.seed = []
        check_seed_deprecation(env)
        env.seed = 123
        check_seed_deprecation(env)
        del env.seed
        check_seed_deprecation(env)
        assert len(caught_warnings) == 0


def test_check_reset_options():
    """Tests the check_reset_options function."""
    with pytest.raises(
        gym.error.Error,
        match=re.escape(
            "The `reset` method does not provide an `options` or `**kwargs` keyword argument"
        ),
    ):
        check_reset_options(GenericTestEnv(reset_func=lambda self: (0, {})))


@pytest.mark.parametrize(
    "test,step_func,message",
    [
        [
            AssertionError,
            lambda self, action: (np.random.normal(), 0, False, False, {}),
            "Deterministic step observations are not equivalent for the same seed and action",
        ],
        [
            AssertionError,
            lambda self, action: (0, np.random.normal(), False, False, {}),
            "Deterministic step rewards are not equivalent for the same seed and action",
        ],
        [
            AssertionError,
            lambda self, action: (0, 0, False, False, {"value": np.random.normal()}),
            "Deterministic step info are not equivalent for the same seed and action",
        ],
    ],
)
def test_check_step_determinism(test, step_func, message: str):
    """Tests the check_step_determinism function."""
    with pytest.raises(test, match=f"^{re.escape(message)}$"):
        check_step_determinism(GenericTestEnv(step_func=step_func))


@pytest.mark.parametrize(
    "env, error_type, message",
    [
        [
            "Error",
            TypeError,
            "The environment must inherit from the gymnasium.Env class, actual class: <class 'str'>. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.",
        ],
        [
            GenericTestEnv(action_space=None),
            AttributeError,
            "The environment must specify an action space. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.",
        ],
        [
            GenericTestEnv(observation_space=None),
            AttributeError,
            "The environment must specify an observation space. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.",
        ],
    ],
)
def test_check_env(env: gym.Env, error_type, message: str):
    """Tests the check_env function works as expected."""
    with pytest.raises(error_type, match=f"^{re.escape(message)}$"):
        check_env(env)
