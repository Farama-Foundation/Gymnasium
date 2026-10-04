"""Tests that the vector wrappers' type parameters mirror the single-environment wrappers'.

The module-level `_check_*` functions are never called; they are checked by `ty`
(see `[tool.ty]` in `pyproject.toml`), where any `type-assertion-failure` is an error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from typing_extensions import assert_type

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.vector import (
    VectorActionWrapper,
    VectorEnv,
    VectorObservationWrapper,
    VectorRewardWrapper,
    VectorWrapper,
)
from gymnasium.vector.vector_env import ArrayType
from gymnasium.wrappers.vector import (
    DictInfoToList,
    HumanRendering,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    RecordVideo,
)
from gymnasium.wrappers.vector.array_conversion import ArrayConversion


class Outer:
    """The type a wrapper exposes to its user."""


class Inner:
    """The type of the wrapped environment."""


def _parameters(cls: type) -> tuple[Any, ...]:
    """Returns the type parameters of a generic class (set at runtime by `Generic`)."""
    return vars(cls)["__parameters__"]


@pytest.mark.parametrize(
    "vector_cls, single_cls",
    [
        (VectorWrapper, gym.Wrapper),
        (VectorObservationWrapper, gym.ObservationWrapper),
        (VectorActionWrapper, gym.ActionWrapper),
        (VectorRewardWrapper, gym.RewardWrapper),
    ],
)
def test_vector_wrapper_parameters_mirror_wrapper(vector_cls, single_cls):
    """Tests that each vector wrapper has the single-env wrapper's type parameters, plus `ArrayType`."""
    assert _parameters(vector_cls) == (*_parameters(single_cls), ArrayType)


def test_vector_wrapper_parameters():
    """Tests the order of the vector wrapper type parameters."""
    assert _parameters(VectorWrapper) == (
        WrapperObsType,
        WrapperActType,
        ObsType,
        ActType,
        ArrayType,
    )
    assert _parameters(ArrayConversion) == (
        WrapperObsType,
        WrapperActType,
        ObsType,
        ActType,
    )


# `VectorWrapper` exposes the outer types, while its `env` has the inner types
def _check_vector_wrapper(
    envs: VectorWrapper[Outer, Outer, Inner, Inner, np.ndarray],
) -> None:
    obs, _ = envs.reset()
    assert_type(obs, Outer)
    obs, rewards, terminations, truncations, _ = envs.step(Outer())
    assert_type(obs, Outer)
    assert_type(rewards, np.ndarray)
    assert_type(terminations, np.ndarray)
    assert_type(truncations, np.ndarray)

    inner_obs, _ = envs.env.reset()
    assert_type(inner_obs, Inner)
    assert_type(envs.env, VectorEnv[Inner, Inner, np.ndarray])


class _ObservationWrapper(VectorObservationWrapper[Outer, Inner, Inner, np.ndarray]):
    def observations(self, observations: Inner) -> Outer:
        return Outer()


def _check_vector_observation_wrapper(
    envs: VectorEnv[Inner, Inner, np.ndarray],
) -> None:
    wrapped = _ObservationWrapper(envs)
    obs, _ = wrapped.reset()
    assert_type(obs, Outer)
    obs, *_ = wrapped.step(Inner())
    assert_type(obs, Outer)


class _ActionWrapper(VectorActionWrapper[Inner, Outer, Inner, np.ndarray]):
    def actions(self, actions: Outer) -> Inner:
        return Inner()


def _check_vector_action_wrapper(envs: VectorEnv[Inner, Inner, np.ndarray]) -> None:
    wrapped = _ActionWrapper(envs)
    obs, *_ = wrapped.step(Outer())
    assert_type(obs, Inner)


class _RewardWrapper(VectorRewardWrapper[Inner, Inner, np.ndarray]):
    def rewards(self, rewards: np.ndarray) -> np.ndarray:
        return rewards


def _check_vector_reward_wrapper(envs: VectorEnv[Inner, Inner, np.ndarray]) -> None:
    wrapped = _RewardWrapper(envs)
    _, rewards, *_ = wrapped.step(Inner())
    assert_type(rewards, np.ndarray)


def _check_array_conversion(
    envs: ArrayConversion[Outer, Outer, Inner, Inner],
) -> None:
    obs, rewards, *_ = envs.step(Outer())
    assert_type(obs, Outer)
    assert_type(rewards, Any)
    inner_obs, _ = envs.env.reset()
    assert_type(inner_obs, Inner)


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        RecordEpisodeStatistics,
        NormalizeReward,
        RecordVideo,
        HumanRendering,
        DictInfoToList,
    ],
)
def test_pass_through_wrapper_parameters(wrapper_cls):
    """Tests that the pass-through vector wrappers are generic in the wrapped environment's types."""
    assert _parameters(wrapper_cls)[:2] == (ObsType, ActType)


# The pass-through wrappers expose the wrapped environment's types
def _check_pass_through_wrappers(
    envs: VectorEnv[Inner, Outer, np.ndarray],
    obs_envs: VectorEnv[np.ndarray, Outer, np.ndarray],
) -> None:
    obs, *_ = RecordEpisodeStatistics(envs).step(Outer())
    assert_type(obs, Inner)
    obs, *_ = NormalizeReward(envs).step(Outer())
    assert_type(obs, Inner)
    obs, *_ = RecordVideo(envs, video_folder="videos").step(Outer())
    assert_type(obs, Inner)
    obs, *_ = HumanRendering(envs).step(Outer())
    assert_type(obs, Inner)
    obs, _ = DictInfoToList(envs).reset()
    assert_type(obs, Inner)

    normalized = NormalizeObservation(obs_envs)
    obs_array, _ = normalized.reset()
    assert_type(obs_array, np.ndarray)
    normalized.step(Outer())
