"""Test suite for vector TransformReward wrapper."""

import numpy as np
import pytest

from gymnasium import wrappers
from gymnasium.vector import AutoresetMode
from tests.wrappers.vector.utils import (
    EpisodeEndTracker,
    check_vector_wrapper_equivalence,
    make_staggered_vector_env,
)


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize("num_envs", (1, 3))
def test_equivalence_with_wrapper(autoreset_mode: AutoresetMode, num_envs: int):
    """Checks equivalence with `TransformReward` applied to each sub-environment for an element-wise func."""
    check_vector_wrapper_equivalence(
        autoreset_mode,
        num_envs,
        "CartPole-v1",
        wrappers.vector.TransformReward,
        {"func": lambda reward: 2 * reward - 0.5},
        wrappers.TransformReward,
        {"func": lambda reward: 2 * reward - 0.5},
    )


@pytest.mark.parametrize("xp_name", ["torch", "jax.numpy"])
def test_next_step_autoreset_converted_rewards(
    xp_name: str,
    env_id: str = "CartPole-v1",
    num_envs: int = 3,
    num_steps: int = 50,
):
    """Reward wrappers assume numpy, therefore, are applied before converting to other array frameworks, e.g., torch or jax."""
    pytest.importorskip("array_api_compat")
    xp = pytest.importorskip(xp_name)
    from gymnasium.wrappers.vector import ArrayConversion

    def func(reward):
        return 2 * reward - 0.5

    numpy_env = wrappers.vector.TransformReward(
        make_staggered_vector_env(env_id, num_envs, AutoresetMode.NEXT_STEP), func
    )
    xp_env = ArrayConversion(
        wrappers.vector.TransformReward(
            make_staggered_vector_env(env_id, num_envs, AutoresetMode.NEXT_STEP),
            func,
        ),
        env_xp=np,
        target_xp=xp,
    )
    numpy_env.reset(seed=123)
    xp_env.reset(seed=123)

    numpy_env.action_space.seed(123)
    tracker = EpisodeEndTracker(num_envs)
    for _ in range(num_steps):
        action = numpy_env.action_space.sample()
        _, numpy_rewards, terminations, truncations, _ = numpy_env.step(action)
        _, xp_rewards, _, _, _ = xp_env.step(xp.asarray(action))

        assert not isinstance(xp_rewards, np.ndarray)
        assert np.allclose(np.asarray(xp_rewards), numpy_rewards)
        tracker.update(terminations, truncations)

    tracker.check(num_steps)
    numpy_env.close()
    xp_env.close()
