"""Test vector environment implementations."""

from functools import partial

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.experimental.vector import (
    AsyncVectorEnv,
    SyncVectorEnv,
    VectorEnv,
    VectorEnvWrapper,
)
from gymnasium.spaces import Discrete, Tuple
from tests.experimental.vector.testing_utils import CustomSpace, make_env
from tests.testing_env import GenericTestEnv


ENV_ID = "CartPole-v1"
NUM_ENVS = 3
ENV_STEPS = 50
SEED = 42


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

    for _ in range(num_steps):
        actions = async_env.action_space.sample()
        assert actions in sync_env.action_space

        # fmt: off
        async_observations, async_rewards, async_terminations, async_truncations, async_infos = async_env.step(actions)
        sync_observations, sync_rewards, sync_terminations, sync_truncations, sync_infos = sync_env.step(actions)
        # fmt: on

        if any(sync_terminations) or any(sync_truncations):
            assert "final_observation" in async_infos
            assert "_final_observation" in async_infos
            assert "final_observation" in sync_infos
            assert "_final_observation" in sync_infos

        assert np.all(async_observations == sync_observations)
        assert np.all(async_rewards == sync_rewards)
        assert np.all(async_terminations == sync_terminations)
        assert np.all(async_truncations == sync_truncations)

    async_env.close()
    sync_env.close()


def test_custom_space_vector_env():
    """Test custom space with vector environment."""
    env = VectorEnv(4, CustomSpace(), CustomSpace())

    assert isinstance(env.single_observation_space, CustomSpace)
    assert isinstance(env.observation_space, Tuple)

    assert isinstance(env.single_action_space, CustomSpace)
    assert isinstance(env.action_space, Tuple)


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
    assert (
        obs == np.array([0])
        and termination == np.array([True])
        and info["reset"] == np.array([True])
    )
    assert "final_observation" in info and "final_info" in info
    assert info["final_observation"] == np.array([0]) and info["final_info"] == {
        "action": 3
    }


@pytest.mark.parametrize("asynchronous", [True, False])
def test_vector_env_info(asynchronous):
    """Test the vector environment info."""
    env = gym.vector.make(
        ENV_ID, num_envs=NUM_ENVS, asynchronous=asynchronous, disable_env_checker=True
    )
    env.reset(seed=SEED)
    for _ in range(ENV_STEPS):
        env.action_space.seed(SEED)
        action = env.action_space.sample()
        _, _, terminations, terminations, infos = env.step(action)
        if any(terminations) or any(terminations):
            assert len(infos["final_observation"]) == NUM_ENVS
            assert len(infos["_final_observation"]) == NUM_ENVS

            assert isinstance(infos["final_observation"], np.ndarray)
            assert isinstance(infos["_final_observation"], np.ndarray)

            for i, (terminated, truncated) in enumerate(
                zip(terminations, terminations)
            ):
                if terminated or truncated:
                    assert infos["_final_observation"][i]
                else:
                    assert not infos["_final_observation"][i]
                    assert infos["final_observation"][i] is None


@pytest.mark.parametrize("concurrent_ends", [1, 2, 3])
def test_vector_env_info_concurrent_terminations(concurrent_ends):
    """Tests the vector environment info with concurrent terminations."""
    # envs that need to terminate together will have the same action
    actions = [0] * concurrent_ends + [1] * (NUM_ENVS - concurrent_ends)
    envs = [make_env(ENV_ID, SEED) for _ in range(NUM_ENVS)]
    envs = SyncVectorEnv(envs)

    for _ in range(ENV_STEPS):
        _, _, terminateds, truncateds, infos = envs.step(actions)
        if any(terminateds) or any(truncateds):
            for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
                if i < concurrent_ends:
                    assert terminated or truncated
                    assert infos["_final_observation"][i]
                else:
                    assert not infos["_final_observation"][i]
                    assert infos["final_observation"][i] is None
            return


class DummyVectorWrapper(VectorEnvWrapper):
    """Dummy Vector wrapper."""

    def __init__(self, env):
        """Initialise the wrapper."""
        self.env = env
        self.counter = 0

    def reset_async(self, **kwargs):
        """Reset the environment."""
        super().reset_async()
        self.counter += 1


def test_vector_env_wrapper_inheritance():
    """Test the vector custom wrapper."""
    env = gym.vector.make("FrozenLake-v1", asynchronous=False)
    wrapped = DummyVectorWrapper(env)
    wrapped.reset()
    assert wrapped.counter == 1


def test_vector_env_wrapper_attributes():
    """Test if `set_attr`, `call` methods for VecEnvWrapper get correctly forwarded to the vector env it is wrapping."""
    env = gym.vector.make("CartPole-v1", num_envs=3)
    wrapped = DummyVectorWrapper(gym.vector.make("CartPole-v1", num_envs=3))

    assert np.allclose(wrapped.call("gravity"), env.call("gravity"))
    env.set_attr("gravity", [20.0, 20.0, 20.0])
    wrapped.set_attr("gravity", [20.0, 20.0, 20.0])
    assert np.allclose(wrapped.get_attr("gravity"), env.get_attr("gravity"))
