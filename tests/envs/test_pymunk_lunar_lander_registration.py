import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import TimeLimit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _use_headless_pygame():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _disable_physics(env):
    env.unwrapped.demo._step_with_powers = lambda action, continuous: (
        env.unwrapped.demo.state(),
        0.0,
        0.0,
    )


def test_lunar_lander_v4_registration_and_time_limit():
    pytest.importorskip("pymunk")
    from gymnasium.envs.pymunk import LunarLander

    env = gym.make("LunarLander-v4")
    observation, _ = env.reset(seed=123)
    _disable_physics(env)

    assert observation in env.observation_space
    assert isinstance(env, TimeLimit)
    assert isinstance(env.unwrapped, LunarLander)
    assert env.spec.id == "LunarLander-v4"
    assert env.spec.entry_point == "gymnasium.envs.pymunk.lunar_lander:LunarLander"
    assert env.spec.max_episode_steps == 1000

    for _ in range(999):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert not truncated

    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated
    assert truncated
    env.close()


def test_lunar_lander_v4_time_limit_can_be_overridden():
    pytest.importorskip("pymunk")

    env = gym.make("LunarLander-v4", max_episode_steps=3)
    env.reset(seed=123)
    _disable_physics(env)

    assert env.spec.max_episode_steps == 3
    for _ in range(2):
        assert env.step(0)[3] is False
    assert env.step(0)[3] is True
    env.close()


def test_lunar_lander_v4_passes_checker_when_made_from_registry():
    pytest.importorskip("pymunk")
    _use_headless_pygame()

    env = gym.make("LunarLander-v4")
    check_env(env.unwrapped)
    env.close()


@pytest.mark.parametrize("env_id", ["LunarLander-v4", "LunarLanderContinuous-v4"])
def test_lunar_lander_v4_action_modes_pass_checker(env_id):
    pytest.importorskip("pymunk")
    _use_headless_pygame()

    env = gym.make(env_id)
    check_env(env.unwrapped)
    env.close()


@pytest.mark.parametrize("env_id", ["LunarLander-v4", "LunarLanderContinuous-v4"])
def test_lunar_lander_v4_action_modes_support_sync_vectorization(env_id):
    pytest.importorskip("pymunk")

    env = gym.make_vec(env_id, num_envs=2, vectorization_mode="sync")
    observation, _ = env.reset(seed=123)

    assert observation.shape == (2, 8)
    assert env.num_envs == 2
    env.close()


@pytest.mark.parametrize(
    ("box2d_id", "pymunk_id", "action"),
    [
        ("LunarLander-v3", "LunarLander-v4", 0),
        (
            "LunarLanderContinuous-v3",
            "LunarLanderContinuous-v4",
            np.zeros(2, dtype=np.float32),
        ),
    ],
)
def test_v3_v4_reset_and_step_info_contracts_match(box2d_id, pymunk_id, action):
    """Registered Box2D and Pymunk environments return empty public info."""
    pytest.importorskip("Box2D")
    pytest.importorskip("pymunk")
    box2d_env = gym.make(box2d_id)
    pymunk_env = gym.make(pymunk_id)
    try:
        _, box2d_reset_info = box2d_env.reset(seed=123)
        _, pymunk_reset_info = pymunk_env.reset(seed=123)
        *_, box2d_step_info = box2d_env.step(action)
        *_, pymunk_step_info = pymunk_env.step(action)

        assert box2d_reset_info == pymunk_reset_info == {}
        assert box2d_step_info == pymunk_step_info == {}
    finally:
        box2d_env.close()
        pymunk_env.close()


def test_lunar_lander_continuous_v4_registration():
    pytest.importorskip("pymunk")
    from gymnasium.envs.pymunk import LunarLander

    env = gym.make("LunarLanderContinuous-v4")

    assert isinstance(env.unwrapped, LunarLander)
    assert env.unwrapped.continuous is True
    assert env.action_space == gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    assert env.spec.entry_point == "gymnasium.envs.pymunk.lunar_lander:LunarLander"
    assert env.spec.max_episode_steps == 1000
    env.close()


def test_lunar_lander_v3_remains_box2d_backed():
    pytest.importorskip("Box2D")
    from gymnasium.envs.box2d.lunar_lander import LunarLander

    with pytest.warns(DeprecationWarning):
        env = gym.make("LunarLander-v3")
    assert isinstance(env.unwrapped, LunarLander)
    assert env.spec.entry_point == "gymnasium.envs.box2d.lunar_lander:LunarLander"
    env.close()


def test_lunar_lander_continuous_v3_remains_box2d_backed():
    pytest.importorskip("Box2D")
    from gymnasium.envs.box2d.lunar_lander import LunarLander

    with pytest.warns(DeprecationWarning):
        env = gym.make("LunarLanderContinuous-v3")
    assert isinstance(env.unwrapped, LunarLander)
    assert env.unwrapped.continuous is True
    assert env.spec.entry_point == "gymnasium.envs.box2d.lunar_lander:LunarLander"
    env.close()


@pytest.mark.parametrize("render_mode", ["human", "rgb_array"])
def test_lunar_lander_v4_pickles_all_constructor_arguments(render_mode):
    pytest.importorskip("pymunk")
    from gymnasium.envs.pymunk import LunarLander

    env = LunarLander(
        render_mode=render_mode,
        continuous=True,
        gravity=-4.0,
        enable_wind=True,
        wind_power=7.0,
        turbulence_power=0.75,
        solver_iterations=30,
    )
    restored = pickle.loads(pickle.dumps(env))

    assert restored.render_mode == render_mode
    assert restored.continuous is True
    assert restored.gravity == -4.0
    assert restored.enable_wind is True
    assert restored.wind_power == 7.0
    assert restored.turbulence_power == 0.75
    assert restored.solver_iterations == 30

    restored.reset(seed=123)
    assert restored.demo.space.iterations == 30


def test_import_gymnasium_and_v4_spec_do_not_require_pymunk():
    code = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "pymunk" or name.startswith("pymunk."):
        raise ImportError("blocked for test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import gymnasium
assert gymnasium.spec("LunarLander-v4").max_episode_steps == 1000
"""

    subprocess.run([sys.executable, "-c", code], check=True, cwd=PROJECT_ROOT)


def test_lunar_lander_v4_reports_missing_pymunk_dependency():
    code = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "pymunk" or name.startswith("pymunk."):
        raise ImportError("blocked for test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
try:
    gym.make("LunarLander-v4")
except DependencyNotInstalled as error:
    assert 'gymnasium[pymunk]' in str(error)
else:
    raise AssertionError("missing Pymunk did not raise DependencyNotInstalled")
"""

    subprocess.run([sys.executable, "-c", code], check=True, cwd=PROJECT_ROOT)


def test_pymunk_extra_and_package_discovery_configuration():
    project_configuration = (PROJECT_ROOT / "pyproject.toml").read_text()

    assert 'pymunk = ["pygame-ce >=2.1.3", "pymunk >=7.0.0"]' in project_configuration
    assert "gymnasium[atari, box2d, classic-control, pymunk," in project_configuration
    assert 'include = ["gymnasium", "gymnasium.*"]' in project_configuration
    assert (PROJECT_ROOT / "gymnasium/envs/pymunk/__init__.py").is_file()
