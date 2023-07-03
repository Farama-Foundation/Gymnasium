import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
from packaging import version

import gymnasium
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Discrete
from gymnasium.wrappers.compatibility import EnvCompatibility, LegacyEnv


try:
    import gym
except ImportError:
    gym = None


try:
    import shimmy
except ImportError:
    shimmy = None


class LegacyEnvExplicit(LegacyEnv, gymnasium.Env):
    """Legacy env that explicitly implements the old API."""

    observation_space = Discrete(1)
    action_space = Discrete(1)
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        pass

    def reset(self):
        return 0

    def step(self, action):
        return 0, 0, False, {}

    def render(self, mode="human"):
        if mode == "human":
            return
        elif mode == "rgb_array":
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        pass

    def seed(self, seed=None):
        pass


class LegacyEnvImplicit(gymnasium.Env):
    """Legacy env that implicitly implements the old API as a protocol."""

    observation_space = Discrete(1)
    action_space = Discrete(1)
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        pass

    def reset(self):  # type: ignore
        return 0  # type: ignore

    def step(self, action: Any) -> Tuple[int, float, bool, Dict]:
        return 0, 0.0, False, {}

    def render(self, mode: Optional[str] = "human") -> Any:
        if mode == "human":
            return
        elif mode == "rgb_array":
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        pass

    def seed(self, seed: Optional[int] = None):
        pass


def test_explicit():
    old_env = LegacyEnvExplicit()
    assert isinstance(old_env, LegacyEnv)
    env = EnvCompatibility(old_env, render_mode="rgb_array")
    assert env.observation_space == Discrete(1)
    assert env.action_space == Discrete(1)
    assert env.reset() == (0, {})
    assert env.reset(seed=0, options={"some": "option"}) == (0, {})
    assert env.step(0) == (0, 0, False, False, {})
    assert env.render().shape == (1, 1, 3)
    env.close()


def test_implicit():
    old_env = LegacyEnvImplicit()
    assert isinstance(old_env, LegacyEnv)
    env = EnvCompatibility(old_env, render_mode="rgb_array")
    assert env.observation_space == Discrete(1)
    assert env.action_space == Discrete(1)
    assert env.reset() == (0, {})
    assert env.reset(seed=0, options={"some": "option"}) == (0, {})
    assert env.step(0) == (0, 0, False, False, {})
    assert env.render().shape == (1, 1, 3)
    env.close()


def test_make_compatibility_in_spec():
    gymnasium.register(
        id="LegacyTestEnv-v0",
        entry_point=LegacyEnvExplicit,
        apply_api_compatibility=True,
    )
    env = gymnasium.make("LegacyTestEnv-v0", render_mode="rgb_array")
    assert env.observation_space == Discrete(1)
    assert env.action_space == Discrete(1)
    assert env.reset() == (0, {})
    assert env.reset(seed=0, options={"some": "option"}) == (0, {})
    assert env.step(0) == (0, 0, False, False, {})
    img = env.render()
    assert isinstance(img, np.ndarray)
    assert img.shape == (1, 1, 3)  # type: ignore
    env.close()
    del gymnasium.envs.registration.registry["LegacyTestEnv-v0"]


def test_make_compatibility_in_make():
    gymnasium.register(id="LegacyTestEnv-v0", entry_point=LegacyEnvExplicit)
    env = gymnasium.make(
        "LegacyTestEnv-v0", apply_api_compatibility=True, render_mode="rgb_array"
    )
    assert env.observation_space == Discrete(1)
    assert env.action_space == Discrete(1)
    assert env.reset() == (0, {})
    assert env.reset(seed=0, options={"some": "option"}) == (0, {})
    assert env.step(0) == (0, 0, False, False, {})
    img = env.render()
    assert isinstance(img, np.ndarray)
    assert img.shape == (1, 1, 3)  # type: ignore
    env.close()
    del gymnasium.envs.registration.registry["LegacyTestEnv-v0"]


@pytest.mark.parametrize(
    "env",
    (
        pytest.param(
            "GymV21Environment-v0",
            marks=pytest.mark.skipif(
                gym is not None
                and (
                    version.parse(gym.version.VERSION) < version.parse("0.21.0")
                    or version.parse(gym.version.VERSION) >= version.parse("0.26.0")
                ),
                reason="Cannot test GymV21Environment-v0 compatibility env with gym < 0.21.0 or gym >= 0.26.0",
            ),
        ),
        pytest.param(
            "GymV26Environment-v0",
            marks=pytest.mark.skipif(
                gym is not None
                and version.parse(gym.version.VERSION) < version.parse("0.26.0"),
                reason="Cannot test GymV26Environment-v0 compatibility env with gym < 0.26.0",
            ),
        ),
    ),
)
def test_shimmy_gym_compatibility(env):
    assert gymnasium.spec(env) is not None

    if shimmy is None:
        with pytest.raises(
            ImportError,
            match=re.escape(
                "To use the gym compatibility environments, run `pip install shimmy[gym-v21]` or `pip install shimmy[gym-v26]`"
            ),
        ):
            gymnasium.make(env)
    elif gym is None:
        with pytest.raises(
            DependencyNotInstalled,
            match=re.escape(
                "No module named 'gym' (Hint: You need to install gym with `pip install gym` to use gym environments"
            ),
        ):
            gymnasium.make(env, env_id="CartPole-v1")
    else:
        gymnasium.make(env, env_id="CartPole-v1")
