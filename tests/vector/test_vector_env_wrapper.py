from __future__ import annotations

from typing import Any

import numpy as np

from gymnasium import make_vec
from gymnasium.core import ObsType
from gymnasium.experimental.vector import VectorWrapper


class DummyWrapper(VectorWrapper):
    def __init__(self, env):
        self.env = env
        self.counter = 0

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.counter += 1
        return super().reset(seed=seed, options=options)


def test_vector_env_wrapper_inheritance():
    env = make_vec("FrozenLake-v1", vectorization_mode="sync")
    wrapped = DummyWrapper(env)
    wrapped.reset()
    assert wrapped.counter == 1


def test_vector_env_wrapper_attributes():
    """Test if `set_attr`, `call` methods for VecEnvWrapper get correctly forwarded to the vector env it is wrapping."""
    env = make_vec("CartPole-v1", num_envs=3)
    wrapped = DummyWrapper(make_vec("CartPole-v1", num_envs=3))

    assert np.allclose(wrapped.call("gravity"), env.call("gravity"))
    env.set_attr("gravity", [20.0, 20.0, 20.0])
    wrapped.set_attr("gravity", [20.0, 20.0, 20.0])
    assert np.allclose(wrapped.get_attr("gravity"), env.get_attr("gravity"))
