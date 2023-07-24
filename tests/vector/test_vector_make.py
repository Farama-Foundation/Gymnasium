import pytest

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import TimeLimit, TransformObservation
from gymnasium.wrappers.env_checker import PassiveEnvChecker
from tests.wrappers.utils import has_wrapper


def test_vector_make_id():
    env = gym.vector.make("CartPole-v1")
    assert isinstance(env, AsyncVectorEnv)
    assert env.num_envs == 1
    env.close()


@pytest.mark.parametrize("num_envs", [1, 3, 10])
def test_vector_make_num_envs(num_envs):
    env = gym.vector.make("CartPole-v1", num_envs=num_envs)
    assert env.num_envs == num_envs
    env.close()


def test_vector_make_asynchronous():
    env = gym.vector.make("CartPole-v1", asynchronous=True)
    assert isinstance(env, AsyncVectorEnv)
    env.close()

    env = gym.vector.make("CartPole-v1", asynchronous=False)
    assert isinstance(env, SyncVectorEnv)
    env.close()


def test_vector_make_wrappers():
    env = gym.vector.make("CartPole-v1", num_envs=2, asynchronous=False)
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

    env = gym.vector.make(
        "CartPole-v1",
        num_envs=2,
        asynchronous=False,
        wrappers=lambda _env: TransformObservation(_env, lambda obs: obs * 2),
    )
    # As asynchronous environment are inaccessible, synchronous vector must be used
    assert isinstance(env, SyncVectorEnv)
    assert all(has_wrapper(sub_env, TransformObservation) for sub_env in env.envs)

    env.close()


def test_vector_make_disable_env_checker():
    # As asynchronous environment are inaccessible, synchronous vector must be used
    env = gym.vector.make("CartPole-v1", num_envs=1, asynchronous=False)
    assert isinstance(env, SyncVectorEnv)
    assert has_wrapper(env.envs[0], PassiveEnvChecker)
    env.close()

    env = gym.vector.make("CartPole-v1", num_envs=5, asynchronous=False)
    assert isinstance(env, SyncVectorEnv)
    assert has_wrapper(env.envs[0], PassiveEnvChecker)
    assert all(
        has_wrapper(env.envs[i], PassiveEnvChecker) is False for i in [1, 2, 3, 4]
    )
    env.close()

    env = gym.vector.make(
        "CartPole-v1", num_envs=3, asynchronous=False, disable_env_checker=True
    )
    assert isinstance(env, SyncVectorEnv)
    assert all(has_wrapper(sub_env, PassiveEnvChecker) is False for sub_env in env.envs)
    env.close()


def test_vector_make_custom_vec_env_old():
    gym.register(
        "CustomVectorizationExample-v0",
        vector_entry_point="tests.envs.registration.utils_envs:CustomVecEnv",
    )

    env = gym.vector.make("CustomVectorizationExample-v0", num_envs=16)
    assert env.__class__.__name__ == "CustomVecEnv"
    assert env.num_envs == 16
    assert env.observation_space.shape == (16, 3)
    assert env.action_space.shape == (16, 3)


def test_vector_make_custom_vec_env_new():
    gym.register(
        "CustomVectorizationExample-v0",
        vector_entry_point="tests.envs.registration.utils_envs:CustomVecEnv",
    )

    env = gym.make_vec("CustomVectorizationExample-v0", num_envs=16)
    assert env.__class__.__name__ == "CustomVecEnv"
    assert env.num_envs == 16
    assert env.observation_space.shape == (16, 3)
    assert env.action_space.shape == (16, 3)