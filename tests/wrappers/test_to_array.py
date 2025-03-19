"""Test suite for ToArray wrapper."""

import importlib
from functools import partial
from itertools import product
from typing import Any, NamedTuple

import numpy as np
import pytest


array_api_compat = pytest.importorskip("array_api_compat")

from array_api_compat import array_namespace, is_array_api_obj  # noqa: E402

from gymnasium.wrappers.to_array import ToArray, module_namespace, to_xp  # noqa: E402
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

installed_modules_combinations = [
    (s, t) for s, t in product(installed_modules, repeat=2) if s != t
]


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
            # Avoid a dependency on array-api-extra
            # Otherwise, we could use xpx.isclose(data_1, data_2, atol=0.00001).all()
            same_device = data_1.device == data_2.device
            a = np.asarray(data_1)
            b = np.asarray(data_2)
            return np.allclose(a, b, atol=0.00001) and same_device
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
    roundtripped_value = to_xp(to_xp(value, xp=target_xp), xp=source_xp)
    assert xp_data_equivalence(roundtripped_value, expected_value)


def reset_func(self, seed=None, options=None, xp=np):
    """A generic array API reset function."""
    return xp.asarray([1.0, 2.0, 3.0]), {"data": xp.asarray([1, 2, 3])}


def step_func(self, action, xp):
    """A generic array API step function."""
    assert isinstance(action, type(xp.zeros(1)))
    return (
        xp.asarray([1, 2, 3]),
        xp.asarray(5.0),
        xp.asarray(True),
        xp.asarray(False),
        {"data": xp.asarray([1.0, 2.0])},
    )


@pytest.mark.parametrize("env_xp, target_xp", installed_modules_combinations)
def test_to_array_wrapper(env_xp, target_xp):
    _reset_func = partial(reset_func, xp=env_xp)
    _step_func = partial(step_func, xp=env_xp)
    env = GenericTestEnv(reset_func=_reset_func, step_func=_step_func)

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
    wrapped_env = ToArray(env, env_xp=env_xp, target_xp=target_xp)
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
