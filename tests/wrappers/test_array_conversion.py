"""Test suite for ArrayConversion wrapper."""

import importlib
import itertools
import pickle
from typing import Any, NamedTuple

import pytest

import gymnasium


array_api_compat = pytest.importorskip("array_api_compat")
array_api_extra = pytest.importorskip("array_api_extra")

from array_api_compat import array_namespace, is_array_api_obj  # noqa: E402

from gymnasium.wrappers import ArrayConversion  # noqa: E402
from gymnasium.wrappers.array_conversion import (  # noqa: E402
    array_conversion,
    module_namespace,
)
from tests.testing_env import GenericTestEnv  # noqa: E402


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


def xp_data_equivalence(data_1, data_2) -> bool:
    """Return if two variables are equivalent that might contain ``torch.Tensor``."""
    if type(data_1) is type(data_2):
        if isinstance(data_1, dict):
            return data_1.keys() == data_2.keys() and all(
                xp_data_equivalence(data_1[k], data_2[k]) for k in data_1.keys()
            )
        elif isinstance(data_1, (tuple, list)):
            return len(data_1) == len(data_2) and all(
                xp_data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2)
            )
        elif is_array_api_obj(data_1):
            xp = array_namespace(data_1)
            return xp.all(array_api_extra.isclose(data_1, data_2, atol=0.00001))
        else:
            return data_1 == data_2
    else:
        return False


class ExampleNamedTuple(NamedTuple):
    a: Any  # Array API compatible object. Does not have proper typing support yet.
    b: Any  # Same as a


def _supports_higher_precision(xp, low_type, high_type):
    """Check if an array module supports higher precision type."""
    return xp.result_type(low_type, high_type) == high_type


# When converting between array modules (source → target → source), we need to ensure that the
# precision used is supported by both modules. If either module only supports 32-bit types, we must
# use the lower precision to account for the conversion during the roundtrip.
def atleast_float32(source_xp, target_xp):
    """Return source_xp.float64 if both modules support it, otherwise source_xp.float32."""
    source_supports_64 = _supports_higher_precision(
        source_xp, source_xp.float32, source_xp.float64
    )
    target_supports_64 = _supports_higher_precision(
        target_xp, target_xp.float32, target_xp.float64
    )
    return (
        source_xp.float64
        if (source_supports_64 and target_supports_64)
        else source_xp.float32
    )


def atleast_int32(source_xp, target_xp):
    """Return source_xp.int64 if both modules support it, otherwise source_xp.int32."""
    source_supports_64 = _supports_higher_precision(
        source_xp, source_xp.int32, source_xp.int64
    )
    target_supports_64 = _supports_higher_precision(
        target_xp, target_xp.int32, target_xp.int64
    )
    return (
        source_xp.int64
        if (source_supports_64 and target_supports_64)
        else source_xp.int32
    )


def value_parametrization():
    for source_xp, target_xp in installed_modules_combinations:
        xp = module_namespace(source_xp)
        source_xp = module_namespace(source_xp)
        target_xp = module_namespace(target_xp)
        for value, expected_value in [
            (2, xp.asarray(2, dtype=atleast_int32(source_xp, target_xp))),
            (
                (3.0, 4),
                (
                    xp.asarray(3.0, dtype=atleast_float32(source_xp, target_xp)),
                    xp.asarray(4, dtype=atleast_int32(source_xp, target_xp)),
                ),
            ),
            (
                [3.0, 4],
                [
                    xp.asarray(3.0, dtype=atleast_float32(source_xp, target_xp)),
                    xp.asarray(4, dtype=atleast_int32(source_xp, target_xp)),
                ],
            ),
            (
                {
                    "a": 6.0,
                    "b": 7,
                },
                {
                    "a": xp.asarray(6.0, dtype=atleast_float32(source_xp, target_xp)),
                    "b": xp.asarray(7, dtype=atleast_int32(source_xp, target_xp)),
                },
            ),
            (xp.asarray(1.0, dtype=xp.float32), xp.asarray(1.0, dtype=xp.float32)),
            (xp.asarray(1.0, dtype=xp.uint8), xp.asarray(1.0, dtype=xp.uint8)),
            (xp.asarray([1, 2], dtype=xp.int32), xp.asarray([1, 2], dtype=xp.int32)),
            (
                xp.asarray([[1.0], [2.0]], dtype=xp.int32),
                xp.asarray([[1.0], [2.0]], dtype=xp.int32),
            ),
            (
                {
                    "a": (
                        1,
                        xp.asarray(2.0, dtype=xp.float32),
                        xp.asarray([3, 4], dtype=xp.int32),
                    ),
                    "b": {"c": 5},
                },
                {
                    "a": (
                        xp.asarray(1, dtype=atleast_int32(source_xp, target_xp)),
                        xp.asarray(2.0, dtype=xp.float32),
                        xp.asarray([3, 4], dtype=xp.int32),
                    ),
                    "b": {
                        "c": xp.asarray(5, dtype=atleast_int32(source_xp, target_xp))
                    },
                },
            ),
            (
                ExampleNamedTuple(
                    a=xp.asarray([1, 2], dtype=xp.int32),
                    b=xp.asarray([1.0, 2.0], dtype=xp.float32),
                ),
                ExampleNamedTuple(
                    a=xp.asarray([1, 2], dtype=xp.int32),
                    b=xp.asarray([1.0, 2.0], dtype=xp.float32),
                ),
            ),
            (None, None),
        ]:
            yield (source_xp, target_xp, value, expected_value)


@pytest.mark.parametrize(
    "source_xp,target_xp,value,expected_value", value_parametrization()
)
def test_roundtripping(source_xp, target_xp, value, expected_value):
    """Test roundtripping between different Array API compatible frameworks."""
    roundtripped_value = array_conversion(
        array_conversion(value, xp=target_xp), xp=source_xp
    )
    assert xp_data_equivalence(roundtripped_value, expected_value)


@pytest.mark.parametrize("env_xp, target_xp", installed_modules_combinations)
def test_array_conversion_wrapper(env_xp, target_xp):
    # Define reset and step functions without partial to avoid pickling issues

    def reset_func(self, seed=None, options=None):
        """A generic array API reset function."""
        return env_xp.asarray([1.0, 2.0, 3.0]), {"data": env_xp.asarray([1, 2, 3])}

    def step_func(self, action):
        """A generic array API step function."""
        assert isinstance(action, type(env_xp.zeros(1)))
        return (
            env_xp.asarray([1, 2, 3]),
            env_xp.asarray(5.0),
            env_xp.asarray(True),
            env_xp.asarray(False),
            {"data": env_xp.asarray([1.0, 2.0])},
        )

    env = GenericTestEnv(reset_func=reset_func, step_func=step_func)

    # Check that the reset and step for env_xp environment are as expected
    obs, info = env.reset()
    # env_xp is automatically converted to the compatible namespace by array_namespace, so we need
    # to check against the compatible namespace of env_xp in array_api_compat
    env_xp_compat = module_namespace(env_xp)
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
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and array_namespace(info["data"]) is target_xp_compat

    # Check that the wrapped environment can render. This implicitly returns None and requires  a
    # None -> None conversion
    wrapped_env.render()

    # Test that the wrapped environment can be pickled
    env = gymnasium.make("CartPole-v1", disable_env_checker=True)
    wrapped_env = ArrayConversion(env, env_xp=env_xp, target_xp=target_xp)
    pkl = pickle.dumps(wrapped_env)
    pickle.loads(pkl)
