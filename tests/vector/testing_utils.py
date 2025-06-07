"""Testing utilitys for `gymnasium.vector`."""

import time

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple


BaseGymSpaces = (Box, Discrete, MultiDiscrete, MultiBinary)


spaces = [
    Box(low=np.array(-1.0), high=np.array(1.0), dtype=np.float64),
    Box(low=np.array([0.0]), high=np.array([10.0]), dtype=np.float64),
    Box(
        low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64
    ),
    Box(
        low=np.array([[-1.0, 0.0], [0.0, -1.0]]), high=np.ones((2, 2)), dtype=np.float64
    ),
    Box(low=0, high=255, shape=(), dtype=np.uint8),
    Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
    Discrete(2),
    Discrete(5, start=-2),
    Tuple((Discrete(3), Discrete(5))),
    Tuple(
        (
            Discrete(7),
            Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64),
        )
    ),
    MultiDiscrete([11, 13, 17]),
    MultiBinary(19),
    Dict(
        {
            "position": Discrete(23),
            "velocity": Box(
                low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64
            ),
        }
    ),
    Dict(
        {
            "position": Dict({"x": Discrete(29), "y": Discrete(31)}),
            "velocity": Tuple(
                (Discrete(37), Box(low=0, high=255, shape=(), dtype=np.uint8))
            ),
        }
    ),
]

HEIGHT, WIDTH = 64, 64


class SlowEnv(gym.Env):
    """A custom slow environment."""

    def __init__(self, slow_reset=0.3):
        """Initialises the environment with a slow reset parameter used in the `step` and `reset` functions."""
        super().__init__()
        self.slow_reset = slow_reset
        self.observation_space = Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8
        )
        self.action_space = Box(low=0.0, high=1.0, shape=(), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment with a time sleep."""
        super().reset(seed=seed)
        if self.slow_reset > 0:
            time.sleep(self.slow_reset)
        return self.observation_space.sample(), {}

    def step(self, action):
        """Steps through the environment with a time sleep."""
        time.sleep(action)
        observation = self.observation_space.sample()
        reward, terminated, truncated = 0.0, False, False
        return observation, reward, terminated, truncated, {}


class CustomSpace(gym.Space):
    """Minimal custom observation space."""

    def sample(self):
        """Generates a sample from the custom space."""
        return self.np_random.integers(0, 10, ())

    def contains(self, x):
        """Check if the element `x` is contained within the space."""
        return 0 <= x <= 10

    def __eq__(self, other):
        """Check if the two spaces are equal."""
        return isinstance(other, CustomSpace)


custom_spaces = [
    CustomSpace(),
    Tuple((CustomSpace(), Box(low=0, high=255, shape=(), dtype=np.uint8))),
]


class CustomSpaceEnv(gym.Env):
    """An environment with custom spaces for observation and action spaces."""

    def __init__(self):
        """Initialise the environment."""
        super().__init__()
        self.observation_space = CustomSpace()
        self.action_space = CustomSpace()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment."""
        super().reset(seed=seed)
        return "reset", {}

    def step(self, action):
        """Steps through the environment."""
        observation = f"step({action:s})"
        reward, terminated, truncated = 0.0, False, False
        return observation, reward, terminated, truncated, {}


def make_env(env_name, seed, **kwargs):
    """Creates an environment."""

    def _make():
        env = gym.make(env_name, disable_env_checker=True, **kwargs)
        env.action_space.seed(seed)
        env.reset(seed=seed)
        return env

    return _make


def make_slow_env(slow_reset, seed):
    """Creates an environment with slow reset."""

    def _make():
        env = SlowEnv(slow_reset=slow_reset)
        env.reset(seed=seed)
        return env

    return _make


def make_custom_space_env(seed):
    """Creates a custom space environment."""

    def _make():
        env = CustomSpaceEnv()
        env.reset(seed=seed)
        return env

    return _make


def assert_rng_equal(rng_1: np.random.Generator, rng_2: np.random.Generator):
    """Tests whether two random number generators are equal."""
    assert rng_1.bit_generator.state == rng_2.bit_generator.state
