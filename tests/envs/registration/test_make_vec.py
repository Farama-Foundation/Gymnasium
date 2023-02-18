"""Testing of the `gym.make_vec` function."""

import pytest

import gymnasium as gym
from gymnasium.experimental import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import TimeLimit, TransformObservation
from tests.wrappers.utils import has_wrapper


def test_make_vec_env_id():
    """Ensure that the `gym.make_vec` creates the right environment."""
    env = gym.make_vec("CartPole-v1")
    assert isinstance(env, AsyncVectorEnv)
    assert env.num_envs == 1
    env.close()


@pytest.mark.parametrize("num_envs", [1, 3, 10])
def test_make_vec_num_envs(num_envs):
    """Test that the `gym.make_vec` num_envs parameter works."""
    env = gym.make_vec("CartPole-v1", num_envs=num_envs)
    assert env.num_envs == num_envs
    env.close()


def test_make_vec_vectorization_mode():
    """Tests the `gym.make_vec` vectorization mode works."""
    env = gym.make_vec("CartPole-v1", vectorization_mode="async")
    assert isinstance(env, AsyncVectorEnv)
    env.close()

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
        wrappers=[lambda _env: TransformObservation(_env, lambda obs: obs * 2)],
    )
    # As asynchronous environment are inaccessible, synchronous vector must be used
    assert isinstance(env, SyncVectorEnv)
    assert all(has_wrapper(sub_env, TransformObservation) for sub_env in env.envs)

    env.close()
