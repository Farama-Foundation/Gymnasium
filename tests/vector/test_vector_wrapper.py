"""Tests the vector wrappers work as expected."""

from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.vector import VectorWrapper


class DummyVectorWrapper(VectorWrapper):
    """Dummy Vector wrapper that contains a counter function to logging the number of times that reset is called."""

    def __init__(self, env):
        """Initialises the wrapper with the environment creating a counter variable."""
        super().__init__(env)

        self.counter = 0

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Updates the ``counter`` each time at ``reset`` is called."""
        self.counter += 1

        return super().reset(seed=seed, options=options)


def test_vector_env_wrapper_inheritance():
    """Test vector environment wrapper inheritance."""
    env = gym.make_vec("FrozenLake-v1", vectorization_mode="sync")
    wrapped = DummyVectorWrapper(env)
    wrapped.reset()
    assert wrapped.counter == 1

    env.close()


def test_vector_env_wrapper_attributes():
    """Test if `set_attr`, `call` methods for VecEnvWrapper get correctly forwarded to the vector env it is wrapping."""
    env = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    wrapped = DummyVectorWrapper(
        gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    )

    assert np.allclose(wrapped.env.call("gravity"), env.call("gravity"))
    env.set_attr("gravity", [20.0, 20.0, 20.0])
    wrapped.env.set_attr("gravity", [20.0, 20.0, 20.0])
    assert np.allclose(wrapped.env.get_attr("gravity"), env.get_attr("gravity"))

    env.close()


def test_vector_env_metadata():
    """Test if `metadata` property for VectorWrapper correctly forwards to the vector env it is wrapping."""
    env = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    wrapped = DummyVectorWrapper(
        gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    )

    assert env.metadata == wrapped.metadata
    env.metadata = {"render_modes": ["rgb_array"]}
    assert env.metadata != wrapped.metadata

    env.close()
