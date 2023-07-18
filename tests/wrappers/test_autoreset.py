"""Tests the gymnasium.wrapper.AutoResetWrapper operates as expected."""
from typing import Generator, Optional

import numpy as np

import gymnasium as gym
from gymnasium.wrappers import AutoresetV0


class DummyResetEnv(gym.Env):
    """A dummy environment which returns ascending numbers starting at `0` when :meth:`self.step()` is called.

    After the second call to :meth:`self.step()` terminated is true.
    Info dicts are also returned containing the same number returned as an observation, accessible via the key "count".
    This environment is provided for the purpose of testing the autoreset wrapper.
    """

    metadata = {}

    def __init__(self):
        """Initialise the DummyResetEnv."""
        self.action_space = gym.spaces.Box(
            low=np.array([0]), high=np.array([2]), dtype=np.int64
        )
        self.observation_space = gym.spaces.Discrete(2)
        self.count = 0

    def step(self, action: int):
        """Steps the DummyEnv with the incremented step, reward and terminated `if self.count > 1` and updated info."""
        self.count += 1
        return (
            np.array([self.count]),  # Obs
            self.count > 2,  # Reward
            self.count > 2,  # Terminated
            False,  # Truncated
            {"count": self.count},  # Info
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the DummyEnv to return the count array and info with count."""
        self.count = 0
        return np.array([self.count]), {"count": self.count}


def unwrap_env(env) -> Generator[gym.Wrapper, None, None]:
    """Unwraps an environment yielding all wrappers around environment."""
    while isinstance(env, gym.Wrapper):
        yield type(env)
        env = env.env


def test_autoreset_wrapper_autoreset():
    """Tests the autoreset wrapper actually automatically resets correctly."""
    env = DummyResetEnv()
    env = AutoresetV0(env)

    obs, info = env.reset()
    assert obs == np.array([0])
    assert info == {"count": 0}

    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert (terminated or truncated) is False
    assert info == {"count": 1}

    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([2])
    assert (terminated or truncated) is False
    assert reward == 0
    assert info == {"count": 2}

    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([0])
    assert (terminated or truncated) is True
    assert reward == 1
    assert info == {
        "count": 0,
        "final_observation": np.array([3]),
        "final_info": {"count": 3},
    }

    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert (terminated or truncated) is False
    assert info == {"count": 1}

    env.close()
