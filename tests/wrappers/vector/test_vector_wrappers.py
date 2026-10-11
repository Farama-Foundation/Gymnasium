"""Tests that the vectorised wrappers operate identically in `VectorEnv(Wrapper)` and `VectorWrapper(VectorEnv)`.

The exception is the data converter wrappers
 * Data conversion wrappers - `JaxToTorch`, `JaxToNumpy` and `NumpyToJax`
 * Normalizing wrappers - `NormalizeObservation` and `NormalizeReward`
 * Different implementations - `LambdaObservation`, `LambdaReward` and `LambdaAction`
 * Different random sources - `StickyAction`
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import wrappers
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.vector.vector_env import AutoresetMode
from tests.testing_env import GenericTestEnv
from tests.wrappers.vector.utils import check_vector_wrapper_equivalence


@pytest.fixture
def custom_environments():
    gym.register(
        "DictObsEnv-v0",
        lambda: GenericTestEnv(
            observation_space=Dict({"a": Box(0, 1), "b": Discrete(5)})
        ),
    )

    yield

    del gym.registry["DictObsEnv-v0"]


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize("num_envs", (1, 3))
@pytest.mark.parametrize(
    "env_id, wrapper_name, kwargs",
    (
        ("DictObsEnv-v0", "FilterObservation", {"filter_keys": ["a"]}),
        ("CartPole-v1", "FlattenObservation", {}),
        ("CarRacing-v3", "GrayscaleObservation", {}),
        ("CarRacing-v3", "ResizeObservation", {"shape": (35, 45)}),
        ("CarRacing-v3", "ReshapeObservation", {"shape": (96, 48, 6)}),
        (
            "CartPole-v1",
            "RescaleObservation",
            {
                "min_obs": np.array([0, -np.inf, 0, -np.inf]),
                "max_obs": np.array([1, np.inf, 1, np.inf]),
            },
        ),
        ("CarRacing-v3", "DtypeObservation", {"dtype": np.int32}),
        # ("CartPole-v1", "RenderObservation", {}),  # not implemented
        # ("CartPole-v1", "TimeAwareObservation", {}),  # not implemented
        # ("CartPole-v1", "FrameStackObservation", {}),  # not implemented
        # ("CartPole-v1", "DelayObservation", {}),  # not implemented
        ("MountainCarContinuous-v0", "ClipAction", {}),
        (
            "MountainCarContinuous-v0",
            "RescaleAction",
            {"min_action": 1, "max_action": 2},
        ),
        ("CartPole-v1", "ClipReward", {"min_reward": -0.25, "max_reward": 0.75}),
    ),
)
def test_vector_wrapper_equivalence(
    autoreset_mode: AutoresetMode,
    num_envs: int,
    env_id: str,
    wrapper_name: str,
    kwargs: dict[str, Any],
    custom_environments,  # pytest fixture
):
    """Checks that `VectorWrapper(VectorEnv)` and `VectorEnv(Wrapper)` are equivalent for every autoreset mode."""
    check_vector_wrapper_equivalence(
        autoreset_mode,
        num_envs,
        env_id,
        getattr(wrappers.vector, wrapper_name),
        kwargs,
        getattr(wrappers, wrapper_name),
        kwargs,
    )
