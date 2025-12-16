"""Test vector environment implementations."""

from __future__ import annotations

import re
from functools import partial

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode
from tests.spaces.utils import TESTING_SPACES, TESTING_SPACES_IDS
from tests.testing_env import GenericTestEnv
from tests.vector.testing_utils import make_env


@pytest.mark.parametrize("shared_memory", [True, False])
@pytest.mark.parametrize(
    "autoreset_mode", [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP]
)
def test_vector_env_equal(shared_memory, autoreset_mode):
    """Test that vector environment are equal for both async and sync variants."""
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]
    num_steps = 100

    async_env = AsyncVectorEnv(
        env_fns, shared_memory=shared_memory, autoreset_mode=autoreset_mode
    )
    sync_env = SyncVectorEnv(env_fns, autoreset_mode=autoreset_mode)

    assert async_env.num_envs == sync_env.num_envs
    assert async_env.observation_space == sync_env.observation_space
    assert async_env.single_observation_space == sync_env.single_observation_space
    assert async_env.action_space == sync_env.action_space
    assert async_env.single_action_space == sync_env.single_action_space

    async_observations, async_infos = async_env.reset(seed=0)
    sync_observations, sync_infos = sync_env.reset(seed=0)
    assert np.all(async_observations == sync_observations)
    assert data_equivalence(async_infos, sync_infos)

    for _ in range(num_steps):
        actions = async_env.action_space.sample()
        assert actions in sync_env.action_space

        (
            async_observations,
            async_rewards,
            async_terminations,
            async_truncations,
            async_infos,
        ) = async_env.step(actions)
        (
            sync_observations,
            sync_rewards,
            sync_terminations,
            sync_truncations,
            sync_infos,
        ) = sync_env.step(actions)

        assert np.all(async_observations == sync_observations)
        assert np.all(async_rewards == sync_rewards)
        assert np.all(async_terminations == sync_terminations)
        assert np.all(async_truncations == sync_truncations)
        assert data_equivalence(async_infos, sync_infos)

    async_env.close()
    sync_env.close()


def debug_step_func(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
    assert action in self.action_space
    return self.observation_space.sample(), 0, False, False, {}


@pytest.mark.parametrize(
    "vectoriser",
    (
        SyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=True),
        partial(AsyncVectorEnv, shared_memory=False),
    ),
    ids=["Sync", "Async with shared memory", "Async without shared memory"],
)
@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_vector_obs_action_spaces(vectoriser, space, num_envs=3):
    try:
        envs = vectoriser(
            [
                lambda: GenericTestEnv(
                    action_space=space,
                    observation_space=space,
                    step_func=debug_step_func,
                )
                for _ in range(num_envs)
            ]
        )
    except TypeError as err:
        assert (
            "has a dynamic shape so its not possible to make a static shared memory."
            in str(err)
        )
        pytest.skip("Skipping space with dynamic shape")

    assert envs.observation_space == envs.action_space

    obs, _ = envs.reset()
    assert obs in envs.observation_space
    obs, _, _, _, _ = envs.step(envs.action_space.sample())

    envs.close()


@pytest.mark.parametrize(
    "vectoriser",
    (
        SyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=True),
        partial(AsyncVectorEnv, shared_memory=False),
    ),
    ids=["Sync", "Async with shared memory", "Async without shared memory"],
)
def test_final_obs_info(vectoriser):
    """Tests that the vector environments correctly return the final observation and info."""

    def reset_fn(self, seed=None, options=None):
        return 0, {"reset": True}

    def thunk():
        return GenericTestEnv(
            action_space=Discrete(4),
            observation_space=Discrete(4),
            reset_func=reset_fn,
            step_func=lambda self, action: (
                action if action < 3 else 0,
                0,
                action >= 3,
                False,
                {"action": action},
            ),
        )

    env = vectoriser([thunk])
    obs, info = env.reset()
    assert obs == np.array([0]) and info == {
        "reset": np.array([True]),
        "_reset": np.array([True]),
    }

    obs, _, termination, _, info = env.step([1])
    assert (
        obs == np.array([1])
        and termination == np.array([False])
        and info == {"action": np.array([1]), "_action": np.array([True])}
    )

    obs, _, termination, _, info = env.step([2])
    assert (
        obs == np.array([2])
        and termination == np.array([False])
        and info == {"action": np.array([2]), "_action": np.array([True])}
    )

    obs, _, termination, _, info = env.step([3])
    assert obs == np.array([0]) and info == {"action": 3, "_action": np.array([True])}

    obs, _, terminated, _, info = env.step([4])
    assert (
        obs == np.array([0])
        and termination == np.array([True])
        and info["reset"] == np.array([True])
    )

    env.close()


@pytest.fixture
def example_env_list():
    """Example vector environment."""
    return [make_env("CartPole-v1", i) for i in range(4)]


@pytest.mark.parametrize(
    "venv_constructor",
    [
        SyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=True),
        partial(AsyncVectorEnv, shared_memory=False),
    ],
)
def test_random_seeding_basics(venv_constructor, example_env_list):
    seed = 42
    vector_env = venv_constructor(example_env_list)
    vector_env.reset(seed=seed)
    assert vector_env.np_random_seed == tuple(
        seed + i for i in range(vector_env.num_envs)
    )
    # resetting with seed=None means seed remains the same
    vector_env.reset(seed=None)
    assert vector_env.np_random_seed == tuple(
        seed + i for i in range(vector_env.num_envs)
    )


@pytest.mark.parametrize(
    "venv_constructor",
    [
        SyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=True),
        partial(AsyncVectorEnv, shared_memory=False),
    ],
)
def test_random_seeds_set_at_retrieval(venv_constructor, example_env_list):
    vector_env = venv_constructor(example_env_list)
    assert len(set(vector_env.np_random_seed)) == vector_env.num_envs
    # default seed starts at zero. Adjust or remove this test if the default seed changes
    assert vector_env.np_random_seed == tuple(range(vector_env.num_envs))


@pytest.mark.parametrize(
    "vectoriser",
    [
        SyncVectorEnv,
        AsyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=False),
    ],
    ids=["Sync", "Async(shared_memory=True)", "Async(shared_memory=False)"],
)
def test_partial_reset(vectoriser):
    envs = vectoriser(
        [lambda: gym.make("CartPole-v1") for _ in range(3)],
        autoreset_mode=AutoresetMode.DISABLED,
    )
    reset_obs, _ = envs.reset(seed=[0, 1, 2])

    envs.action_space.seed(123)
    envs.step(envs.action_space.sample())
    envs.step(envs.action_space.sample())
    step_obs, *_ = envs.step(envs.action_space.sample())

    reset_mask_obs, _ = envs.reset(
        seed=[0, 1, 0], options={"reset_mask": np.array([True, True, False])}
    )
    assert np.all(reset_mask_obs[:2] == reset_obs[:2])
    assert np.all(reset_mask_obs[2] == step_obs[2])

    envs.close()


@pytest.mark.parametrize(
    "vectoriser",
    [
        SyncVectorEnv,
        AsyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=False),
    ],
    ids=["Sync", "Async(shared_memory=True)", "Async(shared_memory=False)"],
)
def test_partial_reset_failure(vectoriser):
    envs = vectoriser(
        [lambda: gym.make("CartPole-v1") for _ in range(3)],
        autoreset_mode=AutoresetMode.DISABLED,
    )

    # Test first reset using a mask
    # with pytest.raises(AssertionError):
    #     envs.reset(options={"reset_mask": np.array([True, True, False])})

    # Reset with all trues
    envs.reset(options={"reset_mask": np.array([True, True, True])})

    # Reset with mask of an incorrect shape
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "`options['reset_mask': mask]` must have shape `(3,)`, got (1,)"
        ),
    ):
        envs.reset(options={"reset_mask": np.array([True])})
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "options['reset_mask': mask]` must have shape `(3,)`, got (4,)"
        ),
    ):
        envs.reset(options={"reset_mask": np.array([True, True, False, False])})
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "`options['reset_mask': mask]` must have shape `(3,)`, got (1, 3)"
        ),
    ):
        envs.reset(options={"reset_mask": np.array([[True, True, True]])})
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "`options['reset_mask': mask]` must contain a boolean array, got reset_mask=[False False False]"
        ),
    ):
        envs.reset(options={"reset_mask": np.array([False, False, False])})
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "`options['reset_mask': mask]` must have `dtype=np.bool_`, got int64"
        ),
    ):
        envs.reset(options={"reset_mask": np.array([1, 1, 0])})
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "`options['reset_mask': mask]` must have `dtype=np.bool_`, got float64"
        ),
    ):
        envs.reset(options={"reset_mask": np.array([1.0, 1.0, 0.0])})


@pytest.mark.parametrize(
    "vectoriser",
    [
        SyncVectorEnv,
        AsyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=False),
    ],
    ids=["Sync", "Async(shared_memory=True)", "Async(shared_memory=False)"],
)
def test_action_count_compatibility(vectoriser):
    """Test that the number of actions is compatible with the number of environments."""
    num_envs = 4
    envs = vectoriser(
        [lambda: gym.make("CartPole-v1") for _ in range(num_envs)],
        autoreset_mode=AutoresetMode.DISABLED,
    )

    # Reset the environment
    envs.reset()

    # Test correct number of actions (should work)
    correct_actions = envs.action_space.sample()
    assert len(correct_actions) == num_envs

    # Test with actions that match the number of environments
    obs, rewards, terminations, truncations, infos = envs.step(correct_actions)
    assert len(obs) == num_envs
    assert len(rewards) == num_envs
    assert len(terminations) == num_envs
    assert len(truncations) == num_envs

    # Test with too few actions (should raise error)
    with pytest.raises(ValueError):
        envs.step(correct_actions[: num_envs - 1])

    # Test with too many actions (should raise error)
    with pytest.raises(ValueError):
        envs.step(np.concatenate([correct_actions, correct_actions[:1]]))

    # Test with scalar action (should raise error for vector env)
    with pytest.raises(TypeError):
        envs.step(0)

    envs.close()
