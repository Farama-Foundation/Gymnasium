"""Testing of the `gym.make_vec` function."""
import re

import pytest

import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.classic_control.cartpole import CartPoleVectorEnv
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import TimeLimit, TransformObservation
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import has_wrapper


def test_make_vec_env_id():
    """Ensure that the `gym.make_vec` creates the right environment."""
    env = gym.make_vec("CartPole-v1")
    assert isinstance(env, CartPoleVectorEnv)
    assert env.num_envs == 1
    env.close()


@pytest.mark.parametrize("num_envs", [1, 3, 10])
@pytest.mark.parametrize("vectorization_mode", ["vector_entry_point", "async", "sync"])
def test_make_vec_num_envs(num_envs, vectorization_mode):
    """Test that the `gym.make_vec` num_envs parameter works."""
    env = gym.make_vec(
        "CartPole-v1", num_envs=num_envs, vectorization_mode=vectorization_mode
    )
    assert env.num_envs == num_envs
    env.close()


def test_make_vec_vectorization_mode():
    """Tests the `gym.make_vec` vectorization mode works."""
    # Test the default value for spec with and without `vector_entry_point`
    env_spec = gym.spec("CartPole-v1")
    assert env_spec is not None and env_spec.vector_entry_point is not None
    env = gym.make_vec("CartPole-v1")
    assert isinstance(env, CartPoleVectorEnv)
    env.close()

    env_spec = gym.spec("Pendulum-v1")
    assert env_spec is not None and env_spec.vector_entry_point is None
    env = gym.make_vec("Pendulum-v1")
    assert isinstance(env, SyncVectorEnv)
    env.close()

    # Test `vector_entry_point`
    env = gym.make_vec("CartPole-v1", vectorization_mode="vector_entry_point")
    assert isinstance(env, CartPoleVectorEnv)
    env.close()

    with pytest.raises(
        gym.error.Error,
        match=re.escape(
            "Cannot create vectorized environment for Pendulum-v1 because it doesn't have a vector entry point defined."
        ),
    ):
        gym.make_vec("Pendulum-v1", vectorization_mode="vector_entry_point")

    # Test `async`
    env = gym.make_vec("CartPole-v1", vectorization_mode="async")
    assert isinstance(env, AsyncVectorEnv)
    env.close()

    gym.register("VecOnlyEnv-v0", vector_entry_point=CartPoleVectorEnv)
    with pytest.raises(
        gym.error.Error,
        match=re.escape(
            "Cannot create vectorized environment for VecOnlyEnv-v0 because it doesn't have an entry point defined."
        ),
    ):
        gym.make_vec("VecOnlyEnv-v0", vectorization_mode="async")
    del gym.registry["VecOnlyEnv-v0"]

    env = gym.make_vec("CartPole-v1", vectorization_mode="sync")
    assert isinstance(env, SyncVectorEnv)
    env.close()


def test_make_vec_wrappers():
    """Tests that the `gym.make_vec` wrappers parameter works."""
    env = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
    assert isinstance(env, SyncVectorEnv)
    assert len(env.envs) == 2

    sub_env = env.envs[0]
    assert isinstance(sub_env, gym.Env)
    assert sub_env.spec is not None
    if sub_env.spec.max_episode_steps is not None:
        assert has_wrapper(sub_env, TimeLimit)

    assert all(
        has_wrapper(sub_env, TransformObservation) is False for sub_env in env.envs
    )
    env.close()

    env = gym.make_vec(
        "CartPole-v1",
        num_envs=2,
        vectorization_mode="sync",
        wrappers=[
            lambda _env: TransformObservation(
                _env, lambda obs: obs * 2, sub_env.observation_space
            )
        ],
    )
    # As asynchronous environment are inaccessible, synchronous vector must be used
    assert isinstance(env, SyncVectorEnv)
    assert all(has_wrapper(sub_env, TransformObservation) for sub_env in env.envs)

    env.close()


@pytest.mark.parametrize(
    "env_id, kwargs",
    (
        ("CartPole-v1", {}),
        ("CartPole-v1", {"num_envs": 3}),
        ("CartPole-v1", {"vectorization_mode": "sync"}),
        ("CartPole-v1", {"vectorization_mode": "vector_entry_point"}),
        (
            "CartPole-v1",
            {"vector_kwargs": {"copy": False}, "vectorization_mode": "sync"},
        ),
        (
            "CartPole-v1",
            {
                "wrappers": (gym.wrappers.TimeAwareObservation,),
                "vectorization_mode": "sync",
            },
        ),
        ("CartPole-v1", {"render_mode": "rgb_array"}),
    ),
)
def test_make_vec_with_spec(env_id: str, kwargs: dict):
    envs = gym.make_vec(env_id, **kwargs)
    assert envs.spec is not None
    recreated_envs = gym.make_vec(envs.spec)

    # Assert equivalence
    assert envs.spec == recreated_envs.spec
    assert envs.num_envs == recreated_envs.num_envs

    assert envs.observation_space == recreated_envs.observation_space
    assert envs.single_observation_space == recreated_envs.single_observation_space
    assert envs.action_space == recreated_envs.action_space
    assert envs.single_action_space == recreated_envs.single_action_space

    assert type(envs) == type(recreated_envs)

    envs.close()
    recreated_envs.close()


def test_async_with_dynamically_registered_env():
    gym.register("TestEnv-v0", CartPoleEnv)

    gym.make_vec("TestEnv-v0", vectorization_mode="async")

    del gym.registry["TestEnv-v0"]


def test_async_registry():
    gym.register("async-registry-test-env-v0", GenericTestEnv)
    gym.make_vec("async-registry-test-env-v0", num_envs=2, vectorization_mode="async")
    del gym.registry["async-registry-test-env-v0"]
