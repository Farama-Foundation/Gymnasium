"""Compare Box2D LunarLander and the experimental Pymunk prototype with PPO.

This script is for early migration discussion only. It does not register the
Pymunk prototype as a public environment and does not imply trajectory parity
between Box2D and Pymunk.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pymunk_lunar_lander_terrain import (  # noqa: E402
    ExperimentalPymunkLunarLanderEnv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--output-csv", type=Path, default=Path("lunar_lander_pymunk_ppo.csv")
    )
    parser.add_argument(
        "--output-png", type=Path, default=Path("lunar_lander_pymunk_ppo.png")
    )
    parser.add_argument("--success-return-threshold", type=float, default=200.0)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


def make_box2d_env():
    """Create the current registered Box2D LunarLander."""
    gym = importlib.import_module("gymnasium")
    return gym.make("LunarLander-v3", disable_env_checker=True)


def make_pymunk_env():
    """Create the private experimental Pymunk LunarLander."""
    return ExperimentalPymunkLunarLanderEnv()


def evaluate_policy(
    model,
    make_env: Callable[[], Any],
    seed: int,
    episodes: int,
    success_return_threshold: float,
) -> tuple[float, float, float, float]:
    """Run deterministic evaluation episodes."""
    returns = []
    lengths = []
    successes = []

    for episode in range(episodes):
        env = make_env()
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        episode_length = 0
        final_info = {}

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, final_info = env.step(action)
            episode_return += float(reward)
            episode_length += 1

            if terminated or truncated:
                break

        env.close()
        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(
            float(
                final_info.get("is_success", episode_return >= success_return_threshold)
            )
        )

    return (
        float(np.mean(returns)),
        float(np.std(returns)),
        float(np.mean(successes)),
        float(np.mean(lengths)),
    )


def train_and_evaluate(
    engine: str,
    make_env: Callable[[], Any],
    seed: int,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    """Train PPO and collect periodic evaluation rows."""
    from stable_baselines3 import PPO

    env = make_env()
    env.reset(seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        verbose=0,
    )

    rows: list[dict[str, float | int | str]] = []
    completed_steps = 0
    evaluation_points = list(
        range(args.eval_freq, args.train_steps + 1, args.eval_freq)
    )
    if not evaluation_points or evaluation_points[-1] != args.train_steps:
        evaluation_points.append(args.train_steps)

    for target_steps in evaluation_points:
        steps_to_train = target_steps - completed_steps
        if steps_to_train > 0:
            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
            completed_steps = target_steps

        mean_return, return_std, success_rate, mean_episode_length = evaluate_policy(
            model,
            make_env,
            seed=seed + 10_000 + target_steps,
            episodes=args.eval_episodes,
            success_return_threshold=args.success_return_threshold,
        )
        rows.append(
            {
                "engine": engine,
                "seed": seed,
                "timestep": target_steps,
                "mean_return": mean_return,
                "return_std": return_std,
                "success_rate": success_rate,
                "mean_episode_length": mean_episode_length,
            }
        )

    env.close()
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_csv: Path) -> None:
    """Write evaluation rows to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "engine",
                "seed",
                "timestep",
                "mean_return",
                "return_std",
                "success_rate",
                "mean_episode_length",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, float | int | str]], output_png: Path) -> None:
    """Write a learning-curve PNG comparing engines."""
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    engines = sorted({str(row["engine"]) for row in rows})

    _, axis = plt.subplots()
    for engine in engines:
        engine_rows = [row for row in rows if row["engine"] == engine]
        timesteps = sorted({int(row["timestep"]) for row in engine_rows})
        means = []
        stds = []
        for timestep in timesteps:
            returns = [
                float(row["mean_return"])
                for row in engine_rows
                if int(row["timestep"]) == timestep
            ]
            means.append(float(np.mean(returns)))
            stds.append(float(np.std(returns)))

        axis.plot(timesteps, means, marker="o", label=engine)
        axis.fill_between(
            timesteps,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
        )

    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Mean deterministic return")
    axis.legend()
    axis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def main() -> None:
    """Run the comparison."""
    args = parse_args()
    engines = {
        "box2d": make_box2d_env,
        "pymunk": make_pymunk_env,
    }

    rows = []
    for seed in args.seeds:
        for engine, make_env in engines.items():
            rows.extend(train_and_evaluate(engine, make_env, seed, args))

    write_csv(rows, args.output_csv)
    write_plot(rows, args.output_png)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
