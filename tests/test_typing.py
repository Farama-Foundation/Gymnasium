"""Pins the variance decisions of `gymnasium.typing`.

The runtime tests check the declared variance of each TypeVar and that modules reuse
(rather than redefine) the public TypeVars.

The module-level `_check_*` functions are never called; they are checked by `ty`
(see `[tool.ty]` in `pyproject.toml`). Positive cases must type-check, and negative
cases carry a `ty: ignore` comment such that, if the variance regresses, the ignore
comment becomes unused and is reported as an error.
"""

from __future__ import annotations

from typing import Any

import pytest
from typing_extensions import assert_type

import gymnasium as gym
import gymnasium.typing as gym_typing
from gymnasium import core
from gymnasium.vector import VectorEnv, vector_env
from gymnasium.wrappers.vector import vectorize_action, vectorize_observation


class Base:
    """A base type for the variance checks."""


class Sub(Base):
    """A subtype of `Base` for the variance checks."""


@pytest.mark.parametrize(
    "name, covariant, contravariant",
    [
        # `Env` is invariant for backwards compatibility
        ("ObsType", False, False),
        ("ActType", False, False),
        ("WrapperObsType", False, False),
        ("WrapperActType", False, False),
        # `VectorEnv` is variance-correct such that PEP 695 inference does not change its semantics
        ("VectorObsType_co", True, False),
        ("VectorActType_contra", False, True),
        ("VectorRewardType_co", True, False),
        ("VectorBoolType_co", True, False),
        ("VectorWrappedObsType", False, False),
        ("VectorWrappedActType", False, False),
        ("VectorWrappedRewardType", False, False),
        ("ArrayType", False, False),
    ],
)
def test_typevar_variance(name: str, covariant: bool, contravariant: bool):
    """Tests the declared variance of the public TypeVars."""
    type_var = getattr(gym_typing, name)
    assert type_var.__name__ == name
    assert type_var.__covariant__ is covariant
    assert type_var.__contravariant__ is contravariant


def test_typevars_are_not_redefined():
    """Tests that modules re-export the public TypeVars rather than declaring duplicates with the same name."""
    assert core.ObsType is gym_typing.ObsType
    assert core.ActType is gym_typing.ActType
    for name in vector_env.__all__:
        if hasattr(gym_typing, name):
            assert getattr(vector_env, name) is getattr(gym_typing, name)
    assert vectorize_observation.VectorWrappedObsType is gym_typing.VectorWrappedObsType
    assert vectorize_action.VectorWrappedActType is gym_typing.VectorWrappedActType


# Observations are covariant
def _check_obs_covariant(
    envs: VectorEnv[Sub, Any, Any, Any],
) -> VectorEnv[Base, Any, Any, Any]:
    return envs


def _check_obs_not_contravariant(
    envs: VectorEnv[Base, Any, Any, Any],
) -> VectorEnv[Sub, Any, Any, Any]:
    return envs  # ty: ignore[invalid-return-type]


# Actions are contravariant
def _check_act_contravariant(
    envs: VectorEnv[Any, Base, Any, Any],
) -> VectorEnv[Any, Sub, Any, Any]:
    return envs


def _check_act_not_covariant(
    envs: VectorEnv[Any, Sub, Any, Any],
) -> VectorEnv[Any, Base, Any, Any]:
    return envs  # ty: ignore[invalid-return-type]


# Rewards are covariant
def _check_reward_covariant(
    envs: VectorEnv[Any, Any, Sub, Any],
) -> VectorEnv[Any, Any, Base, Any]:
    return envs


def _check_reward_not_contravariant(
    envs: VectorEnv[Any, Any, Base, Any],
) -> VectorEnv[Any, Any, Sub, Any]:
    return envs  # ty: ignore[invalid-return-type]


# Terminations and truncations are covariant
def _check_bool_covariant(
    envs: VectorEnv[Any, Any, Any, Sub],
) -> VectorEnv[Any, Any, Any, Base]:
    return envs


def _check_bool_not_contravariant(
    envs: VectorEnv[Any, Any, Any, Base],
) -> VectorEnv[Any, Any, Any, Sub]:
    return envs  # ty: ignore[invalid-return-type]


# `VectorEnv` substitutes its type parameters, falling back to `Any` when omitted
def _check_vector_env_types(envs: VectorEnv[Sub, Base, Sub, Sub]) -> None:
    obs, _ = envs.reset()
    assert_type(obs, Sub)
    obs, rewards, terminations, truncations, _ = envs.step(Base())
    assert_type(obs, Sub)
    assert_type(rewards, Sub)
    assert_type(terminations, Sub)
    assert_type(truncations, Sub)


def _check_vector_env_defaults(envs: VectorEnv) -> None:
    obs, _ = envs.reset()
    assert_type(obs, Any)


# `Env` remains invariant
def _check_env_obs_not_covariant(env: gym.Env[Sub, Any]) -> gym.Env[Base, Any]:
    return env  # ty: ignore[invalid-return-type]


def _check_env_act_not_contravariant(env: gym.Env[Any, Base]) -> gym.Env[Any, Sub]:
    return env  # ty: ignore[invalid-return-type]
