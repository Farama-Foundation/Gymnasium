from functools import partial

import numpy as np
import pytest

from gymnasium.spaces import Discrete
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from tests.testing_env import GenericTestEnv
from tests.vector.utils import make_env


@pytest.mark.parametrize("shared_memory", [True, False])
def test_vector_env_equal(shared_memory):
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

    for _ in range(num_steps):
        actions = async_env.action_space.sample()
        assert actions in sync_env.action_space

        # fmt: off
        async_observations, async_rewards, async_terminateds, async_truncateds, async_infos = async_env.step(actions)
        sync_observations, sync_rewards, sync_terminateds, sync_truncateds, sync_infos = sync_env.step(actions)
        # fmt: on

        if any(sync_terminateds) or any(sync_truncateds):
            assert "final_observation" in async_infos
            assert "_final_observation" in async_infos
            assert "final_observation" in sync_infos
            assert "_final_observation" in sync_infos

        assert np.all(async_observations == sync_observations)
        assert np.all(async_rewards == sync_rewards)
        assert np.all(async_terminateds == sync_terminateds)
        assert np.all(async_truncateds == sync_truncateds)

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
            reset_fn=reset_fn,
            step_fn=lambda self, action: (
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
    assert (
        obs == np.array([0])
        and termination == np.array([True])
        and info["reset"] == np.array([True])
    )
    assert "final_observation" in info and "final_info" in info
    assert info["final_observation"] == np.array([0]) and info["final_info"] == {
        "action": 3
    }


@pytest.mark.parametrize(
    "vectoriser",
    (
        SyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=True),
        partial(AsyncVectorEnv, shared_memory=False),
    ),
    ids=["Sync", "Async with shared memory", "Async without shared memory"],
)
def test_reset_with_list_of_options(vectoriser):
    """Tests that the vector environments can be reset with different options."""

    def reset_fn(self, seed=None, options=None):
        return [0], dict(options=options, **(options or {}))

    def thunk():
        return GenericTestEnv(reset_fn=reset_fn)

    env = vectoriser([thunk, thunk, thunk])
    _, infos = env.reset()
    assert np.array_equal(infos["options"], (None, None, None))

    # Test options broadcasting.
    options = {"arg1": 123, "arg2": "abc", "arg3": True}
    _, infos = env.reset(options=options)
    assert np.array_equal(infos["options"], (options, options, options))
    assert np.array_equal(infos["arg1"], (123, 123, 123))
    assert np.array_equal(infos["arg2"], ("abc", "abc", "abc"))
    assert np.array_equal(infos["arg3"], (True, True, True))

    # When not all options are either list, tuple, or ndarray, default to options broadcasting.
    options = {"arg1": "123", "arg2": ("a", "b", "c")}
    _, infos = env.reset(options=options)
    assert np.array_equal(infos["options"], (options, options, options))
    assert np.array_equal(infos["arg1"], ("123", "123", "123"))
    for value in infos["arg2"]:
        assert value == ("a", "b", "c")

    # When not all options have the same length, default to options broadcasting.
    options = {"arg1": [1, 2], "arg2": [1, 2, 3]}
    _, infos = env.reset(options=options)
    assert np.array_equal(infos["options"], (options, options, options))

    for dtype in (list, tuple, np.array):
        options = {"arg1": dtype([1, 2, 3]), "arg2": [1, 2, 3]}
        _, infos = env.reset(options=options)
        assert np.array_equal(infos["arg1"], options["arg1"])
        assert np.array_equal(infos["arg2"], options["arg2"])

        # When options is omitted (or None), reuse cached options.
        _, infos = env.reset()
        assert np.array_equal(infos["arg1"], options["arg1"])
        assert np.array_equal(infos["arg2"], options["arg2"])

        _, infos = env.reset(options=None)
        assert np.array_equal(infos["arg1"], options["arg1"])
        assert np.array_equal(infos["arg2"], options["arg2"])
