"""Test suite for DictInfoTolist wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers.vector import DictInfoToList


def test_usage_in_vector_env(env_id: str = "CartPole-v1", num_envs: int = 3):
    env = gym.make(env_id, disable_env_checker=True)
    vector_env = gym.make_vec(env_id, num_envs=num_envs)

    DictInfoToList(vector_env)

    with pytest.raises(TypeError):
        DictInfoToList(env)


class ResetOptionAsInfo(VectorEnv):
    """Minimal implementation to test the conversion of vector dict info to list info."""

    def reset(
        self,
        *,
        seed: int | None = None,
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


def _sub_env_info(vector_info: dict[str, Any], i: int) -> dict[str, Any]:
    """Extracts the info of the `i`-th sub-environment from a vector info using the `_key` masks."""
    sub_env_info = {}
    for key, value in vector_info.items():
        if key.startswith("_") or not vector_info[f"_{key}"][i]:
            continue
        sub_env_info[key] = (
            _sub_env_info(value, i) if isinstance(value, dict) else value[i]
        )
    return sub_env_info


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize("num_envs", (1, 3))
def test_autoreset_mode_equivalence(
    autoreset_mode: AutoresetMode,
    num_envs: int,
    env_id: str = "CartPole-v1",
    num_steps: int = 50,
    max_episode_steps: int = 7,
):
    """Checks the list infos match the dict infos for each autoreset mode, including `final_obs` / `final_info` and partial resets."""
    make_kwargs = dict(
        id=env_id,
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": autoreset_mode},
        # Adds nested dict info on episode end
        wrappers=(gym.wrappers.RecordEpisodeStatistics,),
        max_episode_steps=max_episode_steps,
    )
    list_env = DictInfoToList(gym.make_vec(**make_kwargs))
    dict_env = gym.make_vec(**make_kwargs)

    _, list_info = list_env.reset(seed=123)
    _, dict_info = dict_env.reset(seed=123)
    assert isinstance(list_info, list) and len(list_info) == num_envs
    for i in range(num_envs):
        assert data_equivalence(list_info[i], _sub_env_info(dict_info, i))

    list_env.action_space.seed(123)
    num_episode_ends = 0
    for _ in range(num_steps):
        action = list_env.action_space.sample()
        _, _, terminations, truncations, list_info = list_env.step(action)
        _, _, _, _, dict_info = dict_env.step(action)

        dones = np.logical_or(terminations, truncations)
        num_episode_ends += int(np.sum(dones))

        assert isinstance(list_info, list) and len(list_info) == num_envs
        for i in range(num_envs):
            # remove the episode time as it isn't deterministic
            list_sub_info = list_info[i]
            dict_sub_info = _sub_env_info(dict_info, i)
            for sub_info in (
                list_sub_info,
                dict_sub_info,
                list_sub_info.get("final_info", {}),
                dict_sub_info.get("final_info", {}),
            ):
                if "episode" in sub_info:
                    sub_info["episode"].pop("t")
            assert data_equivalence(list_sub_info, dict_sub_info)

            if autoreset_mode == AutoresetMode.SAME_STEP:
                assert ("final_obs" in list_info[i]) == dones[i]
                assert ("final_info" in list_info[i]) == dones[i]

        if autoreset_mode == AutoresetMode.DISABLED and np.any(dones):
            _, list_info = list_env.reset(options={"reset_mask": dones})
            _, dict_info = dict_env.reset(options={"reset_mask": dones})
            for i in range(num_envs):
                assert data_equivalence(list_info[i], _sub_env_info(dict_info, i))

    assert num_episode_ends >= num_envs * (num_steps // (max_episode_steps + 1))

    list_env.close()
    dict_env.close()
