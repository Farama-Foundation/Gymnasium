"""Test suite for ToArray wrapper."""

from typing import NamedTuple
from itertools import product
from functools import partial
from typing import Any

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
array_api_compat = pytest.importorskip("array_api_compat")

from array_api_compat import array_namespace  # noqa: E402
from gymnasium.utils.env_checker import data_equivalence  # noqa: E402
from gymnasium.wrappers.to_array import ToArray, to_xp, module_namespace  # noqa: E402
from tests.testing_env import GenericTestEnv  # noqa: E402


array_api_frameworks = ("jax.numpy", "numpy", "torch", "cupy")
array_api_frameworks_combinations = [
    (s, t) for s, t in product(array_api_frameworks, repeat=2) if s != t
]


class ExampleNamedTuple(NamedTuple):
    a: Any
    b: Any


def value_parametrization():
    for xp, target_xp in array_api_frameworks_combinations:
        for value, expected_value in [
            (1.0, xp.asarray(1.0, dtype=xp.float32)),
            (2, xp.asarray(2, dtype=xp.int32)),
            (
                (3.0, 4),
                (xp.asarray(3.0, dtype=xp.float32), xp.asarray(4, dtype=xp.int32)),
            ),
            (
                [3.0, 4],
                [xp.asarray(3.0, dtype=xp.float32), xp.asarray(4, dtype=xp.int32)],
            ),
            (
                {
                    "a": 6.0,
                    "b": 7,
                },
                {
                    "a": xp.asarray(6.0, dtype=xp.float32),
                    "b": xp.asarray(7, dtype=xp.int32),
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
                        xp.asarray(1, dtype=xp.int32),
                        xp.asarray(2.0, dtype=xp.float32),
                        xp.asarray([3, 4], dtype=xp.int32),
                    ),
                    "b": {"c": xp.asarray(5, dtype=xp.int32)},
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
            yield (value, expected_value, xp, target_xp)


@pytest.mark.parametrize(
    "value,expected_value,source_xp,target_xp",
    value_parametrization(),
)
def test_roundtripping(value, expected_value, source_xp, target_xp):
    """We test numpy -> jax -> numpy as this is direction in the NumpyToJax wrapper.

    Warning: Jax doesn't support float64 out of the box, therefore, we only test float32 in this test.
    """
    source_xp = module_namespace(source_xp)
    target_xp = module_namespace(target_xp)
    roundtripped_value = to_xp(to_xp(value, xp=target_xp), xp=source_xp)
    assert data_equivalence(roundtripped_value, expected_value)


def reset_func(self, seed=None, options=None, xp=np):
    """A jax-based reset function."""
    return xp.asarray([1.0, 2.0, 3.0]), {"data": xp.asarray([1, 2, 3])}


def step_func(self, action, xp):
    """A jax-based step function."""
    assert type(action) is type(xp.zeros(1))
    return (
        xp.asarray([1, 2, 3]),
        xp.asarray(5.0),
        xp.asarray(True),
        xp.asarray(False),
        {"data": xp.asarray([1.0, 2.0])},
    )


@pytest.mark.parametrize(
    "env_xp, target_xp",
    array_api_frameworks_combinations,
)
def test_to_array_wrapper(env_xp, target_xp):
    """Tests the ``ToArray`` wrapper."""
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

    action = target_xp.asarray([1, 2], dtype=np.int32)
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    assert array_namespace(obs) is target_xp_compat
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and array_namespace(info["data"]) is target_xp_compat

    # Check that the wrapped environment can render. This implicitly returns None and requires  a
    # None -> None conversion
    wrapped_env.render()
