from unittest.mock import Mock, call

import pytest

import gymnasium as gym
from gymnasium.utils import performance
from gymnasium.vector import AutoresetMode, SyncVectorEnv


@pytest.mark.parametrize(
    ("autoreset_mode", "expected_steps_per_second"),
    [
        (AutoresetMode.NEXT_STEP, 2),
        (AutoresetMode.SAME_STEP, 4),
        (AutoresetMode.DISABLED, 4),
    ],
)
def test_benchmark_vector_step(autoreset_mode, expected_steps_per_second, monkeypatch):
    env = SyncVectorEnv(
        [lambda: gym.make("CartPole-v1", max_episode_steps=1) for _ in range(2)],
        autoreset_mode=autoreset_mode,
    )
    timestamps = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(performance.time, "time", lambda: next(timestamps))

    try:
        assert (
            performance.benchmark_vector_step(env, target_duration=0, seed=123)
            == expected_steps_per_second
        )
    finally:
        env.close()


def mock_time(monkeypatch):
    timestamps = iter([0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr(performance.time, "time", lambda: next(timestamps, 1.0))


@pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True)])
def test_benchmark_step_resets_finished_env(monkeypatch, terminated, truncated):
    env = Mock()
    env.step.side_effect = [
        (None, 0.0, False, False, {}),
        (None, 0.0, terminated, truncated, {}),
    ]
    mock_time(monkeypatch)

    assert performance.benchmark_step(env, target_duration=0, seed=123) == 2
    assert env.reset.call_args_list == [call(seed=123), call()]
    assert env.action_space.sample.call_count == 3


def test_benchmark_init(monkeypatch):
    envs = [Mock(), Mock()]
    env_lambda = Mock(side_effect=envs)
    mock_time(monkeypatch)

    assert performance.benchmark_init(env_lambda, target_duration=0, seed=123) == 2
    for env in envs:
        env.reset.assert_called_once_with(seed=123)


def test_benchmark_render(monkeypatch):
    env = Mock()
    mock_time(monkeypatch)

    assert performance.benchmark_render(env, target_duration=0) == 2
    assert env.render.call_count == 2
