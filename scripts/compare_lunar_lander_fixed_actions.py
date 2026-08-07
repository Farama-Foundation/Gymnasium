"""Compare Box2D and Pymunk LunarLander under fixed action sequences."""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pymunk_lunar_lander_terrain import (  # noqa: E402
    ExperimentalPymunkLunarLanderEnv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_results/lunar_lander/fixed_action_physics.csv"),
    )
    return parser.parse_args()


def make_box2d_env():
    """Create the registered Box2D LunarLander environment."""
    gym = importlib.import_module("gymnasium")
    return gym.make("LunarLander-v3", disable_env_checker=True)


def make_pymunk_env():
    """Create the experimental Pymunk LunarLander environment."""
    return ExperimentalPymunkLunarLanderEnv()


def repeat_to_length(pattern: list[int], max_steps: int) -> list[int]:
    """Repeat and truncate an action pattern to the requested length."""
    repeats = (max_steps // len(pattern)) + 1
    return (pattern * repeats)[:max_steps]


def fixed_action_sequences(max_steps: int) -> dict[str, list[int]]:
    """Build the fixed action sequences used by the comparison."""
    return {
        "do_nothing": [0] * max_steps,
        "main_engine_only": [2] * max_steps,
        "left_engine_only": [1] * max_steps,
        "right_engine_only": [3] * max_steps,
        "alternating_side_engines": repeat_to_length([1, 3], max_steps),
        "main_engine_pulses": repeat_to_length([2, 0, 0, 0, 0], max_steps),
        "side_then_main": (
            [1, 3] * 25 + [0] * 25 + [2] * 50 + [0] * max(0, max_steps - 125)
        )[:max_steps],
        "mixed_control": repeat_to_length([0, 2, 0, 1, 0, 3], max_steps),
    }


def get_raw_state(env: Any) -> dict[str, float | None]:
    """Best-effort raw state extraction.

    Box2D and Pymunk store raw physics state differently.
    If raw internals are unavailable, return None values.
    """
    unwrapped = getattr(env, "unwrapped", env)

    # Pymunk prototype
    if hasattr(unwrapped, "demo") and unwrapped.demo is not None:
        body = unwrapped.demo.lander_body
        left_leg = unwrapped.demo.left_leg_body
        right_leg = unwrapped.demo.right_leg_body
        return {
            "raw_x": float(body.position.x),
            "raw_y": float(body.position.y),
            "raw_vx": float(body.velocity.x),
            "raw_vy": float(body.velocity.y),
            "raw_angle": float(body.angle),
            "raw_angular_velocity": float(body.angular_velocity),
            "left_leg_angle": float(left_leg.angle),
            "right_leg_angle": float(right_leg.angle),
            "left_leg_angular_velocity": float(left_leg.angular_velocity),
            "right_leg_angular_velocity": float(right_leg.angular_velocity),
        }

    # Box2D LunarLander
    if hasattr(unwrapped, "lander") and unwrapped.lander is not None:
        body = unwrapped.lander
        return {
            "raw_x": float(body.position.x),
            "raw_y": float(body.position.y),
            "raw_vx": float(body.linearVelocity.x),
            "raw_vy": float(body.linearVelocity.y),
            "raw_angle": float(body.angle),
            "raw_angular_velocity": float(body.angularVelocity),
            "left_leg_angle": None,
            "right_leg_angle": None,
            "left_leg_angular_velocity": None,
            "right_leg_angular_velocity": None,
        }

    return {
        "raw_x": None,
        "raw_y": None,
        "raw_vx": None,
        "raw_vy": None,
        "raw_angle": None,
        "raw_angular_velocity": None,
        "left_leg_angle": None,
        "right_leg_angle": None,
        "left_leg_angular_velocity": None,
        "right_leg_angular_velocity": None,
    }


def print_initial_debug(env, engine: str) -> None:
    """Print raw body state immediately after reset, before any action."""
    unwrapped = env.unwrapped

    print(f"\n=== INITIAL DEBUG: {engine} ===")

    if engine == "box2d":
        body = unwrapped.lander
        print("position:", body.position)
        print("velocity:", body.linearVelocity)
        print("angle:", body.angle)
        print("angular_velocity:", body.angularVelocity)
        print("mass:", body.mass)
        print("inertia:", body.inertia)
        print("gravity:", unwrapped.world.gravity)

    elif engine == "pymunk":
        body = unwrapped.demo.lander_body
        left_leg = unwrapped.demo.left_leg_body
        right_leg = unwrapped.demo.right_leg_body
        print("position:", body.position)
        print("velocity:", body.velocity)
        print("angle:", body.angle)
        print("angular_velocity:", body.angular_velocity)
        print("mass:", body.mass)
        print("moment:", body.moment)
        print("gravity:", unwrapped.demo.space.gravity)
        print("space damping:", unwrapped.demo.space.damping)
        print("left_leg position:", left_leg.position)
        print("left_leg angle:", left_leg.angle)
        print("left_leg angular_velocity:", left_leg.angular_velocity)

        print("right_leg position:", right_leg.position)
        print("right_leg angle:", right_leg.angle)
        print("right_leg angular_velocity:", right_leg.angular_velocity)

    print(f"=== END INITIAL DEBUG: {engine} ===\n")


def run_fixed_sequence(
    engine: str,
    env_factory,
    sequence_name: str,
    actions: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    """Run one fixed action sequence and return step diagnostics."""
    env = env_factory()
    env.reset(seed=seed)

    if sequence_name == "do_nothing":
        print_initial_debug(env, engine)

    rows = []

    for step_idx, action in enumerate(actions):
        raw_state_before = get_raw_state(env)

        next_observation, reward, terminated, truncated, info = env.step(action)
        raw_state_after = get_raw_state(env)

        rows.append(
            {
                "engine": engine,
                "sequence": sequence_name,
                "seed": seed,
                "step": step_idx,
                "action": action,
                # Gymnasium observation values. These are normalized LunarLander states.
                "obs_x": float(next_observation[0]),
                "obs_y": float(next_observation[1]),
                "obs_vx": float(next_observation[2]),
                "obs_vy": float(next_observation[3]),
                "obs_angle": float(next_observation[4]),
                "obs_angular_velocity": float(next_observation[5]),
                "obs_left_leg_contact": float(next_observation[6]),
                "obs_right_leg_contact": float(next_observation[7]),
                # Raw physics values, when available.
                "raw_x": raw_state_after["raw_x"],
                "raw_y": raw_state_after["raw_y"],
                "raw_vx": raw_state_after["raw_vx"],
                "raw_vy": raw_state_after["raw_vy"],
                "raw_angle": raw_state_after["raw_angle"],
                "raw_angular_velocity": raw_state_after["raw_angular_velocity"],
                "left_leg_angle": raw_state_after["left_leg_angle"],
                "right_leg_angle": raw_state_after["right_leg_angle"],
                "left_leg_angular_velocity": raw_state_after[
                    "left_leg_angular_velocity"
                ],
                "right_leg_angular_velocity": raw_state_after[
                    "right_leg_angular_velocity"
                ],
                # Useful deltas.
                "delta_raw_x": (
                    None
                    if raw_state_before["raw_x"] is None
                    else raw_state_after["raw_x"] - raw_state_before["raw_x"]
                ),
                "delta_raw_y": (
                    None
                    if raw_state_before["raw_y"] is None
                    else raw_state_after["raw_y"] - raw_state_before["raw_y"]
                ),
                "delta_raw_angle": (
                    None
                    if raw_state_before["raw_angle"] is None
                    else raw_state_after["raw_angle"] - raw_state_before["raw_angle"]
                ),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": info.get("termination_reason", "unknown"),
                "is_success": info.get("is_success", None),
            }
        )

        if terminated or truncated:
            break

    env.close()
    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Write fixed-action diagnostics to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "engine",
        "sequence",
        "seed",
        "step",
        "action",
        "obs_x",
        "obs_y",
        "obs_vx",
        "obs_vy",
        "obs_angle",
        "obs_angular_velocity",
        "obs_left_leg_contact",
        "obs_right_leg_contact",
        "raw_x",
        "raw_y",
        "raw_vx",
        "raw_vy",
        "raw_angle",
        "raw_angular_velocity",
        "left_leg_angle",
        "right_leg_angle",
        "left_leg_angular_velocity",
        "right_leg_angular_velocity",
        "delta_raw_x",
        "delta_raw_y",
        "delta_raw_angle",
        "reward",
        "terminated",
        "truncated",
        "termination_reason",
        "is_success",
    ]

    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run all fixed-action comparisons and write their diagnostics."""
    args = parse_args()

    envs = {
        "box2d": make_box2d_env,
        "pymunk": make_pymunk_env,
    }

    sequences = fixed_action_sequences(args.max_steps)

    rows = []
    for sequence_name, actions in sequences.items():
        for engine, env_factory in envs.items():
            rows.extend(
                run_fixed_sequence(
                    engine=engine,
                    env_factory=env_factory,
                    sequence_name=sequence_name,
                    actions=actions,
                    seed=args.seed,
                )
            )

    write_csv(rows, args.output_csv)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
