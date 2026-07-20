"""Sweep Pymunk solver iterations against matched Box2D trajectories."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pymunk

import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_lunar_lander_angular_dynamics import (  # noqa: E402
    articulated_metrics,
)
from scripts.pymunk_lunar_lander_terrain import (  # noqa: E402
    ExperimentalPymunkLunarLanderEnv,
)

ITERATIONS = (10, 20, 30, 40, 45, 60, 90, 120, 180)
STATE_COMPONENTS = (
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular_velocity",
    "left_leg_angle",
    "left_leg_angular_velocity",
    "right_leg_angle",
    "right_leg_angular_velocity",
    "angular_momentum",
)
REWARD_COMPONENTS = (
    "reward",
    "position_velocity_shaping",
    "angle_shaping",
    "leg_contact_shaping",
    "main_engine_penalty",
    "side_engine_penalty",
    "terminal_reward",
)
SCORE_SCALES = {
    "x": 10.0,
    "y": 10.0,
    "vx": 5.0,
    "vy": 5.0,
    "angle": 1.0,
    "angular_velocity": 2.0,
    "left_leg_angle": 1.0,
    "right_leg_angle": 1.0,
    "left_leg_angular_velocity": 2.0,
    "right_leg_angular_velocity": 2.0,
    "angular_momentum": 5.0,
    "reward": 100.0,
    "position_velocity_shaping": 100.0,
    "angle_shaping": 100.0,
    "leg_contact_shaping": 20.0,
    "main_engine_penalty": 0.3,
    "side_engine_penalty": 0.03,
    "terminal_reward": 100.0,
}


def action_sequences(steps: int, random_seed: int = 2026):
    """Build deterministic open-loop action sequences."""
    random_actions = np.random.default_rng(random_seed).integers(0, 4, size=steps)
    return {
        "no_action": [0] * steps,
        "repeated_main": [2] * steps,
        "repeated_left": [1] * steps,
        "repeated_right": [3] * steps,
        "alternating_side": [1 if index % 2 == 0 else 3 for index in range(steps)],
        "seeded_random": random_actions.tolist(),
        "hover_main_noop": [2 if index % 5 == 0 else 0 for index in range(steps)],
    }


def shaping_parts(observation):
    """Calculate the observation-derived reward shaping terms."""
    return np.array(
        [
            -100.0 * np.hypot(observation[0], observation[1])
            - 100.0 * np.hypot(observation[2], observation[3]),
            -100.0 * abs(observation[4]),
            10.0 * observation[6] + 10.0 * observation[7],
        ]
    )


def rollout(
    engine: str,
    seed: int,
    actions,
    solver_iterations: int = 180,
    motor_max_force: float | None = None,
):
    """Roll out one engine under a fixed action sequence."""
    if engine == "box2d":
        env = gym.make("LunarLander-v3", disable_env_checker=True)
    else:
        env = ExperimentalPymunkLunarLanderEnv(solver_iterations=solver_iterations)
    observation, _ = env.reset(seed=seed)
    if engine == "pymunk" and motor_max_force is not None:
        for constraint in env.demo.space.constraints:
            if isinstance(constraint, pymunk.SimpleMotor):
                constraint.max_force = motor_max_force
    previous = shaping_parts(observation) if engine == "box2d" else None
    cumulative = np.zeros(6)
    rows = []
    finished = False
    for action in actions:
        if finished:
            row = dict(rows[-1])
            row["reward"] = 0.0
            row["motor_impulse"] = 0.0
            row["rotary_limit_impulse"] = 0.0
            rows.append(row)
            continue
        observation, reward, terminated, truncated, _ = env.step(action)
        current = shaping_parts(observation)
        parts = np.zeros(6)
        if terminated:
            parts[5] = reward
        else:
            if previous is not None:
                parts[:3] = current - previous
            parts[3] = -0.30 if action == 2 else 0.0
            parts[4] = -0.03 if action in (1, 3) else 0.0
        cumulative += parts
        previous = current

        if engine == "box2d":
            unwrapped = env.unwrapped
            hull, legs = unwrapped.lander, unwrapped.legs
            bodies = [hull, *legs]
            _, _, momentum, _ = articulated_metrics(
                bodies,
                lambda body: body.worldCenter,
                lambda body: body.linearVelocity,
                lambda body: body.angle,
                lambda body: body.angularVelocity,
            )
            motor_impulse = np.mean(
                [abs(leg.joint.GetMotorTorque(50) / 50) for leg in legs]
            )
            limit_impulse = np.mean(
                [abs(leg.joint.GetReactionTorque(50) / 50) for leg in legs]
            )
            values = [
                hull.position.x,
                hull.position.y,
                hull.linearVelocity.x,
                hull.linearVelocity.y,
                hull.angle,
                hull.angularVelocity,
                legs[0].angle,
                legs[0].angularVelocity,
                legs[1].angle,
                legs[1].angularVelocity,
                momentum,
            ]
        else:
            demo = env.demo
            diagnostics = demo.physics_diagnostics(action)
            hull, legs = demo.lander_body, [demo.left_leg_body, demo.right_leg_body]
            motor_impulse = np.mean(
                [diagnostics["left_motor_impulse"], diagnostics["right_motor_impulse"]]
            )
            limit_impulse = np.mean(
                [
                    diagnostics["left_rotary_limit_impulse"],
                    diagnostics["right_rotary_limit_impulse"],
                ]
            )
            values = [
                hull.position.x,
                hull.position.y,
                hull.velocity.x,
                hull.velocity.y,
                hull.angle,
                hull.angular_velocity,
                legs[0].angle,
                legs[0].angular_velocity,
                legs[1].angle,
                legs[1].angular_velocity,
                demo.articulated_angular_momentum(),
            ]
        row = dict(zip(STATE_COMPONENTS, map(float, values), strict=True))
        row.update(dict(zip(REWARD_COMPONENTS[1:], cumulative, strict=True)))
        row["reward"] = float(reward)
        row["motor_impulse"] = float(motor_impulse)
        row["rotary_limit_impulse"] = float(limit_impulse)
        row["terminated"] = terminated
        row["truncated"] = truncated
        rows.append(row)
        finished = terminated or truncated
    env.close()
    return rows


def rmse(left, right, component):
    """Calculate trajectory RMSE for one component."""
    delta = np.array([row[component] for row in left]) - np.array(
        [row[component] for row in right]
    )
    return float(np.sqrt(np.mean(delta * delta)))


def sweep(
    iterations: Sequence[int] = ITERATIONS,
    seeds: Sequence[int] | None = None,
    steps: int = 200,
    motor_max_force: float | None = None,
):
    """Compare solver settings over matched seeds and action sequences."""
    if seeds is None:
        seeds = range(100, 105)
    sequences = action_sequences(steps)
    box = {
        (seed, name): rollout("box2d", seed, actions)
        for seed in seeds
        for name, actions in sequences.items()
    }
    rows = []
    for iteration in iterations:
        for name, actions in sequences.items():
            component_values = defaultdict(list)
            motor_impulses = []
            limit_impulses = []
            for seed in seeds:
                pymunk_rows = rollout(
                    "pymunk", seed, actions, iteration, motor_max_force
                )
                box_rows = box[(seed, name)]
                for component in STATE_COMPONENTS + REWARD_COMPONENTS:
                    component_values[component].append(
                        rmse(box_rows, pymunk_rows, component)
                    )
                motor_impulses.extend(row["motor_impulse"] for row in pymunk_rows)
                limit_impulses.extend(
                    row["rotary_limit_impulse"] for row in pymunk_rows
                )
            result = {"iterations": iteration, "sequence": name}
            for component in STATE_COMPONENTS + REWARD_COMPONENTS:
                result[f"rmse_{component}"] = float(
                    np.mean(component_values[component])
                )
            result["combined_score"] = float(
                np.mean(
                    [
                        result[f"rmse_{component}"] / SCORE_SCALES[component]
                        for component in STATE_COMPONENTS + REWARD_COMPONENTS
                    ]
                )
            )
            result["mean_motor_impulse"] = float(np.mean(motor_impulses))
            result["mean_rotary_limit_impulse"] = float(np.mean(limit_impulses))
            rows.append(result)
    return rows


def aggregate_scores(rows):
    """Average combined scores by solver iteration count."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["iterations"])].append(float(row["combined_score"]))
    return {iteration: float(np.mean(scores)) for iteration, scores in grouped.items()}


def write_rows(path: Path, rows):
    """Write sweep result rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run the solver-iteration sweep from command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, nargs="+", default=list(ITERATIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 105)))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/lunar_lander/solver_iteration_sweep.csv"),
    )
    args = parser.parse_args()
    rows = sweep(args.iterations, args.seeds, args.steps)
    write_rows(args.output, rows)
    for iteration, score in sorted(aggregate_scores(rows).items()):
        print(f"iterations={iteration} combined_score={score:.6f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
