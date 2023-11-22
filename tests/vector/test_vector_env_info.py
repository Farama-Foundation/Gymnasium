"""Test the vector environment information."""
import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import VectorEnv
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from tests.vector.testing_utils import make_env


def test_examples():
    env = VectorEnv()

    # Test num-envs==1 then expand_dims(sub-env-info) == vector-infos
    env.num_envs = 1
    sub_env_info = {"a": 0, "b": 0.0, "c": None, "d": np.zeros((2,)), "e": Discrete(1)}
    vector_infos = env._add_info({}, sub_env_info, 0)
    expected_vector_infos = {
        "a": np.array([0]),
        "b": np.array([0.0]),
        "c": np.array([None], dtype=object),
        "d": np.zeros(
            (
                1,
                2,
            )
        ),
        "e": np.array([Discrete(1)], dtype=object),
        "_a": np.array([True]),
        "_b": np.array([True]),
        "_c": np.array([True]),
        "_d": np.array([True]),
        "_e": np.array([True]),
    }
    assert data_equivalence(vector_infos, expected_vector_infos)

    # Thought: num-envs>1 then vector-infos should have the same structure as sub-env-info
    env.num_envs = 3
    sub_env_infos = [
        {"a": 0, "b": 0.0, "c": None, "d": np.zeros((2,)), "e": Discrete(1)},
        {"a": 1, "b": 1.0, "c": None, "d": np.zeros((2,)), "e": Discrete(2)},
        {"a": 2, "b": 2.0, "c": None, "d": np.zeros((2,)), "e": Discrete(3)},
    ]

    vector_infos = {}
    for i, info in enumerate(sub_env_infos):
        vector_infos = env._add_info(vector_infos, info, i)

    expected_vector_infos = {
        "a": np.array([0, 1, 2]),
        "b": np.array([0.0, 1.0, 2.0]),
        "c": np.array([None, None, None], dtype=object),
        "d": np.zeros((3, 2)),
        "e": np.array([Discrete(1), Discrete(2), Discrete(3)], dtype=object),
        "_a": np.array([True, True, True]),
        "_b": np.array([True, True, True]),
        "_c": np.array([True, True, True]),
        "_d": np.array([True, True, True]),
        "_e": np.array([True, True, True]),
    }
    assert data_equivalence(vector_infos, expected_vector_infos)

    # Test different structures of sub-infos
    env.num_envs = 3
    sub_env_infos = [
        {"a": 1, "b": 1.0},
        {"c": None, "d": np.zeros((2,))},
        {"e": Discrete(3)},
    ]

    vector_infos = {}
    for i, info in enumerate(sub_env_infos):
        vector_infos = env._add_info(vector_infos, info, i)

    expected_vector_infos = {
        "a": np.array([1, 0, 0]),
        "b": np.array([1.0, 0.0, 0.0]),
        "c": np.array([None, None, None], dtype=object),
        "d": np.zeros((3, 2)),
        "e": np.array([None, None, Discrete(3)], dtype=object),
        "_a": np.array([True, False, False]),
        "_b": np.array([True, False, False]),
        "_c": np.array([False, True, False]),
        "_d": np.array([False, True, False]),
        "_e": np.array([False, False, True]),
    }
    assert data_equivalence(vector_infos, expected_vector_infos)

    # Test recursive structure
    env.num_envs = 3
    sub_env_infos = [
        {"episode": {"a": 1, "b": 1.0}},
        {"episode": {"a": 2, "b": 2.0}, "a": 1},
        {"a": 2},
    ]

    vector_infos = {}
    for i, info in enumerate(sub_env_infos):
        vector_infos = env._add_info(vector_infos, info, i)

    expected_vector_infos = {
        "episode": {
            "a": np.array([1, 2, 0]),
            "b": np.array([1.0, 2.0, 0.0]),
            "_a": np.array([True, True, False]),
            "_b": np.array([True, True, False]),
        },
        "_episode": np.array([True, True, False]),
        "a": np.array([0, 1, 2]),
        "_a": np.array([False, True, True]),
    }
    assert data_equivalence(vector_infos, expected_vector_infos)


@pytest.mark.parametrize("vectorization_mode", ["async", "sync"])
def test_vector_env_info(
    vectorization_mode: str,
    env_id: str = "CartPole-v1",
    num_envs: int = 3,
    env_steps: int = 50,
    seed: int = 123,
):
    """Test vector environment info for different vectorization modes."""
    env = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
    )
    env.reset(seed=seed)
    for _ in range(env_steps):
        env.action_space.seed(seed)
        action = env.action_space.sample()
        _, _, terminations, truncations, infos = env.step(action)
        if any(terminations) or any(truncations):
            assert len(infos["final_observation"]) == num_envs
            assert len(infos["_final_observation"]) == num_envs

            assert isinstance(infos["final_observation"], np.ndarray)
            assert isinstance(infos["_final_observation"], np.ndarray)

            for i, (terminated, truncated) in enumerate(zip(terminations, truncations)):
                if terminated or truncated:
                    assert infos["_final_observation"][i]
                else:
                    assert not infos["_final_observation"][i]
                    assert infos["final_observation"][i] is None


@pytest.mark.parametrize("concurrent_ends", [1, 2, 3])
def test_vector_env_info_concurrent_termination(
    concurrent_ends: int,
    env_id: str = "CartPole-v1",
    num_envs: int = 3,
    env_steps: int = 50,
    seed: int = 123,
):
    """Test the vector environment information works with concurrent termination."""
    # envs that need to terminate together will have the same action
    actions = [0] * concurrent_ends + [1] * (num_envs - concurrent_ends)
    envs = [make_env(env_id, seed) for _ in range(num_envs)]
    envs = SyncVectorEnv(envs)

    for _ in range(env_steps):
        _, _, terminations, truncations, infos = envs.step(actions)
        if any(terminations) or any(truncations):
            for i, (terminated, truncated) in enumerate(zip(terminations, truncations)):
                if i < concurrent_ends:
                    assert terminated or truncated
                    assert infos["_final_observation"][i]
                else:
                    assert not infos["_final_observation"][i]
                    assert infos["final_observation"][i] is None
            return
