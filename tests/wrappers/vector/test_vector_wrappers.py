"""Tests that the vectorised wrappers operate identically in `VectorEnv(Wrapper)` and `VectorWrapper(VectorEnv)`.

The exception is the data converter wrappers
 * Data conversion wrappers - `JaxToTorch`, `JaxToNumpy` and `NumpyToJax`
 * Normalizing wrappers - `NormalizeObservation` and `NormalizeReward`
 * Different implementations - `LambdaObservation`, `LambdaReward` and `LambdaAction`
 * Different random sources - `StickyAction`
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import wrappers
from gymnasium.spaces import Box, Dict, Discrete, Tuple
from gymnasium.vector.vector_env import AutoresetMode
from tests.testing_env import GenericTestEnv
from tests.wrappers.vector.utils import check_vector_wrapper_equivalence


@pytest.mark.parametrize("read_only", [False, True])
def test_nested_dict_transform_preserves_input_batch(read_only):
    """Swapping nested fields must not alias or overwrite the source vector batch."""
    space = Dict({key: Dict({"value": Box(0, 1, (2,))}) for key in ("left", "right")})
    env = gym.vector.SyncVectorEnv(
        [lambda: GenericTestEnv(observation_space=space)] * 3
    )
    wrapped = wrappers.vector.VectorizeTransformObservation(
        env,
        wrapper=wrappers.TransformObservation,
        func=lambda obs: {"left": obs["right"], "right": obs["left"]},
        observation_space=space,
    )
    try:
        batch, _ = env.reset(seed=123)
        original = deepcopy(batch)
        if read_only:
            for value in batch.values():
                value["value"].flags.writeable = False
        transformed = wrapped.observations(batch)
        for key, other in [("left", "right"), ("right", "left")]:
            np.testing.assert_array_equal(
                transformed[key]["value"], original[other]["value"]
            )
            np.testing.assert_array_equal(batch[key]["value"], original[key]["value"])
            assert not np.shares_memory(
                transformed[key]["value"], batch[other]["value"]
            )
    finally:
        wrapped.close()


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


@pytest.mark.parametrize("vectorization_mode", ["sync", "async"])
@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize("filter_keys", [(0, 1), (1, 0)])
def test_filter_tuple_observation_equivalence(
    vectorization_mode, autoreset_mode, filter_keys
):
    """Filtering equal tuple subspaces must preserve each field's values."""
    check_vector_wrapper_equivalence(
        autoreset_mode,
        3,
        lambda: GenericTestEnv(
            observation_space=Tuple((Box(0, 1, (2,)), Box(0, 1, (2,))))
        ),
        wrappers.vector.FilterObservation,
        {"filter_keys": filter_keys},
        wrappers.FilterObservation,
        {"filter_keys": filter_keys},
        vectorization_mode=vectorization_mode,
    )
