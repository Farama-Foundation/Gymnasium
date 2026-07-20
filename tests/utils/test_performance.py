import gymnasium as gym
from gymnasium.utils import performance
from gymnasium.vector import AutoresetMode, SyncVectorEnv


def test_benchmark_step_vector_excludes_next_step_autoresets(monkeypatch):
    env = SyncVectorEnv(
        [lambda: gym.make("CartPole-v1", max_episode_steps=1) for _ in range(2)],
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )
    timestamps = iter([0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr(performance.time, "time", lambda: next(timestamps))

    try:
        assert performance.benchmark_step_vector(env, target_duration=0, seed=123) == 2
    finally:
        env.close()
