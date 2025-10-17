"""Test suite for DictInfoTolist wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.vector import DictInfoToList


def test_usage_in_vector_env(env_id: str = "CartPole-v1", num_envs: int = 3):
    env = gym.make(env_id, disable_env_checker=True)
    vector_env = gym.make_vec(env_id, num_envs=num_envs)

    DictInfoToList(vector_env)

    with pytest.raises(AssertionError):
        DictInfoToList(env)


class ResetOptionAsInfo(VectorEnv):
    """Minimal implementation to test the conversion of vector dict info to list info."""

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,  # options are passed are the info output
    ) -> tuple[ObsType, dict[str, Any]]:
        return None, options


def test_update_info():
    env = DictInfoToList(ResetOptionAsInfo())

    # Test num-envs==1 then expand_dims(sub-env-info) == vector-infos
    env.unwrapped.num_envs = 1

    vector_infos = {
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
    _, list_info = env.reset(options=vector_infos)

    # The return dtype of np.array([0]) is platform dependent
    np_array_int_default_dtype = np.array([0]).dtype.type

    expected_list_info = [
        {
            "a": np_array_int_default_dtype(0),
            "b": np.float64(0.0),
            "c": None,
            "d": np.zeros((2,)),
            "e": Discrete(1),
        }
    ]

    assert data_equivalence(list_info, expected_list_info)

    # Thought: num-envs>1 then vector-infos should have the same structure as sub-env-info
    env.unwrapped.num_envs = 3

    vector_infos = {
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
    _, list_info = env.reset(options=vector_infos)
    expected_list_info = [
        {
            "a": np_array_int_default_dtype(0),
            "b": np.float64(0.0),
            "c": None,
            "d": np.zeros((2,)),
            "e": Discrete(1),
        },
        {
            "a": np_array_int_default_dtype(1),
            "b": np.float64(1.0),
            "c": None,
            "d": np.zeros((2,)),
            "e": Discrete(2),
        },
        {
            "a": np_array_int_default_dtype(2),
            "b": np.float64(2.0),
            "c": None,
            "d": np.zeros((2,)),
            "e": Discrete(3),
        },
    ]

    assert list_info[0].keys() == expected_list_info[0].keys()
    for key in list_info[0].keys():
        assert data_equivalence(list_info[0][key], expected_list_info[0][key])
    assert data_equivalence(list_info, expected_list_info)

    # Test different structures of sub-infos
    env.unwrapped.num_envs = 3

    vector_infos = {
        "a": np.array([1, 0, 0]),
        "_a": np.array([True, False, False]),
        "b": np.array([1.0, 0.0, 0.0]),
        "_b": np.array([True, False, False]),
        "c": np.array([None, None, None], dtype=object),
        "_c": np.array([False, True, False]),
        "_d": np.array([False, True, False]),
        "d": np.zeros((3, 2)),
        "e": np.array([None, None, Discrete(3)], dtype=object),
        "_e": np.array([False, False, True]),
    }
    _, list_info = env.reset(options=vector_infos)
    expected_list_info = [
        {"a": np_array_int_default_dtype(1), "b": np.float64(1.0)},
        {"c": None, "d": np.zeros((2,))},
        {"e": Discrete(3)},
    ]
    assert data_equivalence(list_info, expected_list_info)

    # Test recursive structure
    env.unwrapped.num_envs = 3

    vector_infos = {
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
    _, list_info = env.reset(options=vector_infos)
    expected_list_info = [
        {"episode": {"a": np_array_int_default_dtype(1), "b": np.float64(1.0)}},
        {
            "episode": {"a": np_array_int_default_dtype(2), "b": np.float64(2.0)},
            "a": np_array_int_default_dtype(1),
        },
        {"a": np_array_int_default_dtype(2)},
    ]
    assert data_equivalence(list_info, expected_list_info)

    # Test without binary array (can happen with custom vector environment)
    vector_infos = {
        "episode": {
            "a": np.array([1, 2, 0]),
            "b": np.array([1.0, 2.0, 0.0]),
        },
        "a": np.array([0, 1, 2]),
    }
    _, list_info = env.reset(options=vector_infos)
    expected_list_info = [
        {
            "episode": {"a": np_array_int_default_dtype(1), "b": np.float64(1.0)},
            "a": np_array_int_default_dtype(0),
        },
        {
            "episode": {"a": np_array_int_default_dtype(2), "b": np.float64(2.0)},
            "a": np_array_int_default_dtype(1),
        },
        {
            "episode": {"a": np_array_int_default_dtype(0), "b": np.float64(0.0)},
            "a": np_array_int_default_dtype(2),
        },
    ]
    assert data_equivalence(list_info, expected_list_info)


@pytest.mark.parametrize(
    "vector_infos",
    [
        {
            "a": np.array([1, 2]),
            "_a": np.array([True, True, False]),
        },
        {
            "a": np.array([1, 2]),
        },
        {
            "episode": {
                "a": np.array([1, 2]),
                "_a": np.array([True, True, False]),
            },
            "a": np.array([0, 1, 2]),
            "_a": np.array([False, True, True]),
        },
        {
            "episode": {
                "a": np.array([1, 2]),
            },
            "a": np.array([0, 1, 2]),
        },
        {
            "episode": {
                "a": np.array([1, 2, 0]),
                "_a": np.array([True, True, False]),
            },
            "a": np.array([1, 2]),
            "_a": np.array([True, True]),
        },
        {
            "episode": {
                "a": np.array([1, 2, 0]),
            },
            "a": np.array([1, 2]),
        },
    ],
)
def test_errors(vector_infos):
    env = DictInfoToList(ResetOptionAsInfo())
    env.unwrapped.num_envs = 3

    with pytest.raises(AssertionError):
        env.reset(options=vector_infos)
