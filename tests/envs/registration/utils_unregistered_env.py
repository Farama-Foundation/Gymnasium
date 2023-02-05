"""This utility file contains an environment that is registered upon loading the file."""
import gymnasium as gym


class RegisterDuringMakeEnv(gym.Env):
    """Used in `test_registration.py` to check if `env.make` can import and register an env"""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)


gym.register(id="RegisterDuringMake-v0", entry_point=RegisterDuringMakeEnv)
