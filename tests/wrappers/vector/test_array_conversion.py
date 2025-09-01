"""Test suite for vector ArrayConversion wrapper."""

import importlib
import itertools
from functools import partial

import numpy as np
import pytest

from tests.testing_env import GenericTestVectorEnv


array_api_compat = pytest.importorskip("array_api_compat")
from array_api_compat import array_namespace  # noqa: E402


jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
from gymnasium.wrappers.array_conversion import module_namespace  # noqa: E402
from gymnasium.wrappers.vector import ArrayConversion  # noqa: E402
from gymnasium.wrappers.vector import JaxToNumpy  # noqa: E402
from gymnasium.wrappers.vector import JaxToTorch  # noqa: E402
from gymnasium.wrappers.vector import NumpyToTorch  # noqa: E402


# Define available modules
installed_modules = []
array_api_modules = [
    "numpy",
    "jax.numpy",
    "torch",
    "cupy",
    "dask.array",
    "sparse",
    "array_api_strict",
]
for module in array_api_modules:
    try:
        installed_modules.append(importlib.import_module(module))
    except ImportError:
        pass  # Modules that are not installed are skipped

installed_modules_combinations = list(itertools.permutations(installed_modules, 2))


def create_vector_env(env_xp):
    _reset_func = partial(reset_func, num_envs=3, xp=env_xp)
    _step_func = partial(step_func, num_envs=3, xp=env_xp)
    return GenericTestVectorEnv(reset_func=_reset_func, step_func=_step_func)


def reset_func(self, seed=None, options=None, num_envs: int = 1, xp=np):
    return xp.asarray([[1.0, 2.0, 3.0] * num_envs]), {
        "data": xp.asarray([[1, 2, 3] * num_envs])
    }


def step_func(self, action, num_envs: int = 1, xp=np):
    assert isinstance(action, type(xp.zeros(1)))
    return (
        xp.asarray([[1, 2, 3] * num_envs]),
        xp.asarray([5.0] * num_envs),
        xp.asarray([False] * num_envs),
        xp.asarray([False] * num_envs),
        {"data": xp.asarray([[1.0, 2.0] * num_envs])},
    )


@pytest.mark.parametrize("env_xp, target_xp", installed_modules_combinations)
def test_array_conversion_wrapper(env_xp, target_xp):
    env_xp_compat = module_namespace(env_xp)
    env = create_vector_env(env_xp_compat)

    # Check that the reset and step for env_xp environment are as expected
    obs, info = env.reset()
    # env_xp is automatically converted to the compatible namespace by array_namespace, so we need
    # to check against the compatible namespace of env_xp in array_api_compat
    assert array_namespace(obs) is env_xp_compat
    assert isinstance(info, dict) and array_namespace(info["data"]) is env_xp_compat

    obs, reward, terminated, truncated, info = env.step(env_xp_compat.asarray([1, 2]))
    assert array_namespace(obs) is env_xp_compat
    assert array_namespace(reward) is env_xp_compat
    assert array_namespace(terminated) is env_xp_compat
    assert array_namespace(truncated) is env_xp_compat
    assert isinstance(info, dict) and array_namespace(info["data"]) is env_xp_compat

    # Check that the wrapped version is correct.
    target_xp_compat = module_namespace(target_xp)
    wrapped_env = ArrayConversion(env, env_xp=env_xp, target_xp=target_xp)
    obs, info = wrapped_env.reset()
    assert array_namespace(obs) is target_xp_compat
    assert isinstance(info, dict) and array_namespace(info["data"]) is target_xp_compat

    action = target_xp.asarray([1, 2], dtype=target_xp.int32)
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    assert array_namespace(obs) is target_xp_compat
    assert array_namespace(reward) is target_xp_compat
    assert array_namespace(terminated) is target_xp_compat
    assert terminated.dtype == target_xp.bool
    assert array_namespace(truncated) is target_xp_compat
    assert truncated.dtype == target_xp.bool
    assert isinstance(info, dict) and array_namespace(info["data"]) is target_xp_compat

    # Check that the wrapped environment can render. This implicitly returns None and requires  a
    # None -> None conversion
    wrapped_env.render()


@pytest.mark.parametrize("wrapper", [JaxToNumpy, JaxToTorch, NumpyToTorch])
def test_specialized_wrappers(wrapper: type[JaxToNumpy | JaxToTorch | NumpyToTorch]):
    if wrapper is JaxToNumpy:
        env_xp, target_xp = jax.numpy, np
    elif wrapper is JaxToTorch:
        env_xp, target_xp = jax.numpy, torch
    elif wrapper is NumpyToTorch:
        env_xp, target_xp = np, torch
    else:
        raise TypeError(f"Unknown specialized conversion wrapper {type(wrapper)}")
    env_xp_compat = module_namespace(env_xp)
    target_xp_compat = module_namespace(target_xp)

    # The unwrapped test env sanity check is already covered by test_array_conversion_wrapper for
    # all known frameworks, including the specialized ones.
    env = create_vector_env(env_xp_compat)

    # Check that the wrapped version is correct.
    wrapped_env = wrapper(env)
    obs, info = wrapped_env.reset()
    assert array_namespace(obs) is target_xp_compat
    assert isinstance(info, dict) and array_namespace(info["data"]) is target_xp_compat

    action = target_xp.asarray([1, 2], dtype=target_xp.int32)
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    assert array_namespace(obs) is target_xp_compat
    assert array_namespace(reward) is target_xp_compat
    assert array_namespace(terminated) is target_xp_compat
    assert terminated.dtype == target_xp.bool
    assert array_namespace(truncated) is target_xp_compat
    assert truncated.dtype == target_xp.bool
    assert isinstance(info, dict) and array_namespace(info["data"]) is target_xp_compat

    # Check that the wrapped environment can render. This implicitly returns None and requires  a
    # None -> None conversion
    wrapped_env.render()
