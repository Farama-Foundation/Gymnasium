"""Test suite for vector TransformAction wrapper."""

import numpy as np
import pytest

from gymnasium import spaces, wrappers
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from tests.testing_env import GenericTestEnv
from tests.wrappers.vector.test_vector_wrappers import (
    check_vector_wrapper_equivalence,
)


def create_env():
    return GenericTestEnv(
        action_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32),
            high=np.array([10, -5, 10], dtype=np.float32),
        )
    )


def test_action_space_from_single_action_space(
    n_envs: int = 5,
):
    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    vec_env = wrappers.vector.TransformAction(
        vec_env,
        func=lambda x: x + 100,
        single_action_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32) + 100,
            high=np.array([10, -5, 10], dtype=np.float32) + 100,
        ),
    )

    # Check action space
    assert isinstance(vec_env.action_space, spaces.Box)
    assert vec_env.action_space.shape == (n_envs, 3)
    assert vec_env.action_space.dtype == np.float32
    assert (
        vec_env.action_space.low == np.array([[100, 90, 95]] * n_envs, dtype=np.float32)
    ).all()
    assert (
        vec_env.action_space.high
        == np.array([[110, 95, 110]] * n_envs, dtype=np.float32)
    ).all()

    # Check single action space
    assert isinstance(vec_env.single_action_space, spaces.Box)
    assert vec_env.single_action_space.shape == (3,)
    assert vec_env.single_action_space.dtype == np.float32
    assert (
        vec_env.single_action_space.low == np.array([100, 90, 95], dtype=np.float32)
    ).all()
    assert (
        vec_env.single_action_space.high == np.array([110, 95, 110], dtype=np.float32)
    ).all()


def test_warning_on_mismatched_single_action_space(
    n_envs: int = 2,
):
    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    # We only specify action_space without single_action_space, so single_action_space inherits its value from the wrapped env which would not match. This mismatch should give us a warning.
    with pytest.warns(
        Warning,
        match=r"the action space and the batched single action space don't match as expected",
    ):
        vec_env = wrappers.vector.TransformAction(
            vec_env,
            func=lambda x: x + 100,
            action_space=spaces.Box(
                low=np.array([[0, -10, -5]] * n_envs, dtype=np.float32) + 100,
                high=np.array([[10, -5, 10]] * n_envs, dtype=np.float32) + 100,
            ),
        )


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize("num_envs", (1, 3))
def test_equivalence_with_wrapper(autoreset_mode: AutoresetMode, num_envs: int):
    """Checks equivalence with `TransformAction` applied to each sub-environment for an element-wise func."""
    check_vector_wrapper_equivalence(
        autoreset_mode,
        num_envs,
        "MountainCarContinuous-v0",
        wrappers.vector.TransformAction,
        {"func": lambda action: 0.5 * action},
        wrappers.TransformAction,
        {"func": lambda action: 0.5 * action, "action_space": None},
    )
