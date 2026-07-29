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
