import pytest

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import PassiveEnvCheckerV0, TimeLimitV0, LambdaObservationV0
from tests.wrappers.utils import has_wrapper


def test_vector_make_id():
    env = gym.make_vec("CartPole-v1")
    assert isinstance(env, AsyncVectorEnv)
    assert env.num_envs == 1
    env.close()


@pytest.mark.parametrize("num_envs", [1, 3, 10])
def test_vector_make_num_envs(num_envs):
    env = gym.make_vec("CartPole-v1", num_envs=num_envs)
    assert env.num_envs == num_envs
    env.close()


def test_vector_make_asynchronous():
    env = gym.make_vec("CartPole-v1", vectorization_mode="async")
    assert isinstance(env, AsyncVectorEnv)
    env.close()

    env = gym.make_vec("CartPole-v1", vectorization_mode="sync")
    assert isinstance(env, SyncVectorEnv)
    env.close()


def test_vector_make_wrappers():
    env = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
    assert isinstance(env, SyncVectorEnv)
    assert len(env.envs) == 2

    sub_env = env.envs[0]
    assert isinstance(sub_env, gym.Env)
    assert sub_env.spec is not None
    if sub_env.spec.max_episode_steps is not None:
        assert has_wrapper(sub_env, TimeLimitV0)

    assert all(
        has_wrapper(sub_env, LambdaObservationV0) is False for sub_env in env.envs
    )
    env.close()

    env = gym.make_vec(
        "CartPole-v1",
        num_envs=2,
        vectorization_mode="sync",
        wrappers=(lambda _env: LambdaObservationV0(_env, lambda obs: obs * 2, Box(0, 2)),),
    )
    # As asynchronous environment are inaccessible, synchronous vector must be used
    assert isinstance(env, SyncVectorEnv)
    assert all(has_wrapper(sub_env, LambdaObservationV0) for sub_env in env.envs)

    env.close()


@pytest.mark.skip(reason="disable_env_checker not part of `make_vec`")
def test_vector_make_disable_env_checker():
    # As asynchronous environment are inaccessible, synchronous vector must be used
    env = gym.make_vec("CartPole-v1", num_envs=1, asynchronous=False)
    assert isinstance(env, SyncVectorEnv)
    assert has_wrapper(env.envs[0], PassiveEnvCheckerV0)
    env.close()

    env = gym.make_vec("CartPole-v1", num_envs=5, asynchronous=False)
    assert isinstance(env, SyncVectorEnv)
    assert has_wrapper(env.envs[0], PassiveEnvCheckerV0)
    assert all(
        has_wrapper(env.envs[i], PassiveEnvCheckerV0) is False for i in [1, 2, 3, 4]
    )
    env.close()

    env = gym.make_vec(
        "CartPole-v1", num_envs=3, asynchronous=False, disable_env_checker=True
    )
    assert isinstance(env, SyncVectorEnv)
    assert all(
        has_wrapper(sub_env, PassiveEnvCheckerV0) is False for sub_env in env.envs
    )
    env.close()
