from typing import Any, Tuple, Dict, Optional

from gymnasium.core import ActType, ObsType

import gymnasium as gym
from tests.generic_test_env import GenericTestEnv


class RegisterDuringMakeEnv(GenericTestEnv):
    """Used in `test_registration.py` to check if `env.make` can import and register an env"""


class ArgumentEnv(GenericTestEnv):
    def __init__(self, arg1: Any, arg2: Any, arg3: Any):
        super().__init__()

        self.arg1, self.arg2, self.arg3 = arg1, arg2, arg3


# Environments to test render_mode
class NoHuman(GenericTestEnv):
    """Environment that does not have human-rendering."""

    metadata = {"render_modes": ["rgb_array_list"], "render_fps": 4}

    def __init__(self, render_mode: str = None):
        assert render_mode in self.metadata["render_modes"]
        super().__init__(render_mode=render_mode)


class NoHumanOldAPI(gym.Env):
    """Environment that does not have human-rendering."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self):
        pass

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        return self.observation_space.sample(), 0, False, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        return self.observation_space.sample(), {}


class NoHumanNoRGB(gym.Env):
    """Environment that has neither human- nor rgb-rendering"""

    metadata = {"render_modes": ["ascii"], "render_fps": 4}

    def __init__(self, render_mode=None):
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
