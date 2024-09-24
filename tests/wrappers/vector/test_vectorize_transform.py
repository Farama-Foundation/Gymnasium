from functools import partial

import numpy as np

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from tests.testing_env import GenericTestEnv


def test_vectorize_box_to_dict_action():
    def func(x):
        return x["key"]

    envs = SyncVectorEnv([lambda: GenericTestEnv() for _ in range(2)])
    envs = gym.wrappers.vector.VectorizeTransformAction(
        env=envs,
        wrapper=gym.wrappers.TransformAction,
        func=func,
        action_space=gym.spaces.Dict(
            {"key": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)}
        ),
    )

    obs, _ = envs.reset()
    obs, _, _, _, _ = envs.step(envs.action_space.sample())
    envs.close()


def test_vectorize_dict_to_box_obs():
    wrappers = [
        partial(
            gym.wrappers.TransformObservation,
            func=lambda x: {"key1": x[0:1], "key2": x[1:]},
            observation_space=gym.spaces.Dict(
                {
                    "key1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                    "key2": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                }
            ),
        )
    ]
    envs = gym.make_vec(
        "CartPole-v1",
        num_envs=2,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        wrappers=wrappers,
    )
    obs, _ = envs.reset()
    assert obs in envs.observation_space
    obs, _, _, _, _ = envs.step(envs.action_space.sample())
    assert obs in envs.observation_space
    envs.close()
