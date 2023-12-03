"""Test vector environment implementations."""

from functools import partial

import numpy as np
import pytest

from gymnasium.spaces import Discrete
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from tests.testing_env import GenericTestEnv
from tests.vector.testing_utils import make_env


@pytest.mark.parametrize("shared_memory", [True, False])
def test_vector_env_equal(shared_memory):
    """Test that vector environment are equal for both async and sync variants."""
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]
    num_steps = 100

    async_env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    sync_env = SyncVectorEnv(env_fns)

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
