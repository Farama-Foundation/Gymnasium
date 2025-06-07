import pickle
import re
import warnings

import pytest

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import check_env, data_equivalence
from tests.envs.utils import all_testing_env_specs, all_testing_initialised_envs


# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered envs.
PASSIVE_CHECK_IGNORE_WARNING = [
    r"\x1b\[33mWARN: The environment (.*?) is out of date\. You should consider upgrading to version `v(\d)`\.\x1b\[0m",
]


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]


@pytest.mark.parametrize(
    "spec",
    all_testing_env_specs,
    ids=[spec.id for spec in all_testing_env_specs],
)
def test_all_env_api(spec):
    """Check that all environments pass the environment checker with no warnings other than the expected."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = spec.make().unwrapped

        check_env(env, skip_render_check=True)

        env.close()

    for warning in caught_warnings:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_all_env_passive_env_checker(spec):
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = gym.make(spec.id)
        env.reset()
        env.step(env.action_space.sample())

        env.close()

    passive_check_pattern = re.compile("|".join(PASSIVE_CHECK_IGNORE_WARNING))

    for warning in caught_warnings:
        if not passive_check_pattern.search(str(warning.message)):
            raise ValueError(f"Unexpected warning: {warning.message}")


# Note that this precludes running this test in multiple threads.
# However, we probably already can't do multithreading due to some environments.
SEED = 0
NUM_STEPS = 50


@pytest.mark.parametrize(
    "env_spec",
    all_testing_env_specs,
    ids=[env.id for env in all_testing_env_specs],
)
def test_env_determinism_rollout(env_spec: EnvSpec):
    """Run a rollout with two environments and assert equality.

    This test run a rollout of NUM_STEPS steps with two environments
    initialized with the same seed and assert that:

    - observation after first reset are the same
    - same actions are sampled by the two envs
    - observations are contained in the observation space
    - obs, rew, done and info are equals between the two envs
    """
    # Don't check rollout equality if it's a nondeterministic environment.
    if env_spec.nondeterministic is True:
        pytest.skip(f"Skipping {env_spec.id} as it is non-deterministic")

    env_1 = env_spec.make(disable_env_checker=True)
    env_2 = env_spec.make(disable_env_checker=True)

    if env_1.metadata.get("jax", False):
        env_1 = gym.wrappers.JaxToNumpy(env_1)
        env_2 = gym.wrappers.JaxToNumpy(env_2)

    initial_obs_1, initial_info_1 = env_1.reset(seed=SEED)
    initial_obs_2, initial_info_2 = env_2.reset(seed=SEED)
    assert data_equivalence(initial_obs_1, initial_obs_2, exact=True)

    env_1.action_space.seed(SEED)

    for time_step in range(NUM_STEPS):
        # We don't evaluate the determinism of actions
        action = env_1.action_space.sample()

        obs_1, rew_1, terminated_1, truncated_1, info_1 = env_1.step(action)
        obs_2, rew_2, terminated_2, truncated_2, info_2 = env_2.step(action)

        assert data_equivalence(
            obs_1, obs_2, exact=True
        ), f"[{time_step}] obs_1={obs_1}, obs_2={obs_2}"
        assert env_1.observation_space.contains(
            obs_1
        )  # obs_2 verified by previous assertion

        assert rew_1 == rew_2, f"[{time_step}] reward 1={rew_1}, reward 2={rew_2}"
        assert (
            terminated_1 == terminated_2
        ), f"[{time_step}] done 1={terminated_1}, done 2={terminated_2}"
        assert (
            truncated_1 == truncated_2
        ), f"[{time_step}] done 1={truncated_1}, done 2={truncated_2}"
        assert data_equivalence(
            info_1, info_2, exact=True
        ), f"[{time_step}] info_1={info_1}, info_2={info_2}"

        if (
            terminated_1 or truncated_1
        ):  # terminated_2, truncated_2 verified by previous assertion
            env_1.reset(seed=SEED)
            env_2.reset(seed=SEED)

    env_1.close()
    env_2.close()


@pytest.mark.parametrize(
    "env",
    all_testing_initialised_envs,
    ids=[env.spec.id for env in all_testing_initialised_envs if env.spec is not None],
)
def test_pickle_env(env: gym.Env):
    if env.metadata.get("jax", False):
        env = gym.wrappers.JaxToNumpy(env)

    action = env.action_space.sample()

    env_reset = env.reset(seed=123)
    env_step = env.step(action)

    pickled_env = pickle.loads(pickle.dumps(env))
    pickle_reset = pickled_env.reset(seed=123)
    pickle_step = pickled_env.step(action)

    assert data_equivalence(env_reset, pickle_reset)
    assert data_equivalence(env_step, pickle_step)

    env.close()
    pickled_env.close()
