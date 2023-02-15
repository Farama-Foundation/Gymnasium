import numpy as np

import gymnasium as gym
from gymnasium.experimental.vector import VectorWrapper


class DummyWrapper(VectorWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.counter = 0

    def reset(self, **kwargs):
        super().reset()
        self.counter += 1


def test_vector_env_wrapper_inheritance():
    env = gym.make_vec("FrozenLake-v1", vectorization_mode="async")
    wrapped = DummyWrapper(env)
    wrapped.reset()
    assert wrapped.counter == 1


def test_vector_env_wrapper_attributes():
    """Test if `set_attr`, `call` methods for VecEnvWrapper get correctly forwarded to the vector env it is wrapping."""
    env = gym.make_vec("CartPole-v1", num_envs=3)
    wrapped = DummyWrapper(gym.make_vec("CartPole-v1", num_envs=3))

    assert np.allclose(wrapped.call("gravity"), env.call("gravity"))
    env.set_attr("gravity", [20.0, 20.0, 20.0])
    wrapped.set_attr("gravity", [20.0, 20.0, 20.0])
    assert np.allclose(wrapped.get_attr("gravity"), env.get_attr("gravity"))
