"""Tests the vector wrappers work as expected."""
import numpy as np

import gymnasium as gym
from gymnasium.experimental.vector import VectorWrapper


class DummyVectorWrapper(VectorWrapper):
    """Dummy Vector wrapper that contains a counter function to logging the number of times that reset is called."""

    def __init__(self, env):
        """Initialises the wrapper with the environment creating a counter variable."""
        super().__init__(env)
        self.env = env
        self.counter = 0

    def reset(self, **kwargs):
        """Updates the ``counter`` each time at ``reset`` is called."""
        super().reset()
        self.counter += 1


def test_vector_env_wrapper_inheritance():
    """Test vector environment wrapper inheritance."""
    env = gym.make_vec("FrozenLake-v1", vectorization_mode="async")
    wrapped = DummyVectorWrapper(env)
    wrapped.reset()
    assert wrapped.counter == 1


def test_vector_env_wrapper_attributes():
    """Test if `set_attr`, `call` methods for VecEnvWrapper get correctly forwarded to the vector env it is wrapping."""
    env = gym.make_vec("CartPole-v1", num_envs=3)
    wrapped = DummyVectorWrapper(gym.make_vec("CartPole-v1", num_envs=3))

    assert np.allclose(wrapped.call("gravity"), env.call("gravity"))
    env.set_attr("gravity", [20.0, 20.0, 20.0])
    wrapped.set_attr("gravity", [20.0, 20.0, 20.0])
    assert np.allclose(wrapped.get_attr("gravity"), env.get_attr("gravity"))
