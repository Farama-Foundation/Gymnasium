"""Audit Box2D and Pymunk LunarLander angular dynamics without training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium.envs.box2d import lunar_lander as box_module  # noqa: E402
from scripts.pymunk_lunar_lander_terrain import (  # noqa: E402
    ExperimentalPymunkLunarLanderEnv,
    body_center_of_mass_world,
    body_origin_world,
)

SEQUENCES = {
    "no_action": lambda steps: [0] * steps,
    "single_left": lambda steps: [1] + [0] * (steps - 1),
    "single_right": lambda steps: [3] + [0] * (steps - 1),
    "repeated_left": lambda steps: [1] * steps,
    "repeated_right": lambda steps: [3] * steps,
    "alternating_side": lambda steps: [
        1 if step % 2 == 0 else 3 for step in range(steps)
    ],
    "repeated_main": lambda steps: [2] * steps,
}


class RecordingRng:
    """Delegate RNG calls while retaining values sampled by Box2D step()."""

    def __init__(self, generator):
        """Wrap a random generator and initialize sample storage."""
        self.generator = generator
        self.uniform_values: list[float] = []

    def uniform(self, *args, **kwargs):
        """Sample uniformly and record scalar results."""
        value = self.generator.uniform(*args, **kwargs)
        if np.ndim(value) == 0:
            self.uniform_values.append(float(value))
        return value

    def __getattr__(self, name):
        """Delegate other attributes to the wrapped generator."""
        return getattr(self.generator, name)


def articulated_metrics(bodies, position, velocity, angle, angular_velocity):
    """Calculate center, inertia, angular momentum, and mass for bodies."""
    masses = np.array([float(body.mass) for body in bodies])
    positions = np.array([position(body) for body in bodies], dtype=float)
    center = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
    inertia = sum(
        float(body.inertia if hasattr(body, "inertia") else body.moment)
        + float(body.mass) * float(np.sum((positions[index] - center) ** 2))
        for index, body in enumerate(bodies)
    )
    momentum = sum(
        float(body.inertia if hasattr(body, "inertia") else body.moment)
        * angular_velocity(body)
        + (positions[index, 0] - center[0]) * masses[index] * float(velocity(body)[1])
        - (positions[index, 1] - center[1]) * masses[index] * float(velocity(body)[0])
        for index, body in enumerate(bodies)
    )
    return center, float(inertia), float(momentum), float(np.sum(masses))


def box_impulse(action, angle, position, dispersion):
    """Reconstruct a Box2D engine impulse and its application point."""
    tip = np.array([math.sin(angle), math.cos(angle)])
    side = np.array([-tip[1], tip[0]])
    if action == 2:
        offset = (
            tip
            * (box_module.MAIN_ENGINE_Y_LOCATION / box_module.SCALE + 2 * dispersion[0])
            + side * dispersion[1]
        )
        offset[1] *= -1
        impulse = -offset * box_module.MAIN_ENGINE_POWER
    elif action in (1, 3):
        direction = action - 2
        ox = tip[0] * dispersion[0] + side[0] * (
            3 * dispersion[1]
            + direction * box_module.SIDE_ENGINE_AWAY / box_module.SCALE
        )
        oy = -tip[1] * dispersion[0] - side[1] * (
            3 * dispersion[1]
            + direction * box_module.SIDE_ENGINE_AWAY / box_module.SCALE
        )
        offset = np.array(
            [
                ox - tip[0] * 17 / box_module.SCALE,
                oy + tip[1] * box_module.SIDE_ENGINE_HEIGHT / box_module.SCALE,
            ]
        )
        impulse = -np.array([ox, oy]) * box_module.SIDE_ENGINE_POWER
    else:
        return np.zeros(2), np.zeros(2), np.asarray(position)
    return offset, impulse, np.asarray(position) + offset


def initial_rows(seed_count: int):
    """Collect matched initial articulated-state rows across seeds."""
    rows = []
    box_env = gym.make("LunarLander-v3", disable_env_checker=True)
    pymunk_env = ExperimentalPymunkLunarLanderEnv()
    for seed in range(seed_count):
        box_env.reset(seed=seed)
        pymunk_env.reset(seed=seed)
        for engine, bodies, position, velocity, angle, angular_velocity in [
            (
                "box2d",
                [box_env.unwrapped.lander, *box_env.unwrapped.legs],
                lambda body: body.worldCenter,
                lambda body: body.linearVelocity,
                lambda body: body.angle,
                lambda body: body.angularVelocity,
            ),
            (
                "pymunk",
                [
                    pymunk_env.demo.lander_body,
                    pymunk_env.demo.left_leg_body,
                    pymunk_env.demo.right_leg_body,
                ],
                body_center_of_mass_world,
                lambda body: body.velocity,
                lambda body: body.angle,
                lambda body: body.angular_velocity,
            ),
        ]:
            center, inertia, momentum, mass = articulated_metrics(
                bodies, position, velocity, angle, angular_velocity
            )
            hull = bodies[0]
            rows.append(
                {
                    "engine": engine,
                    "seed": seed,
                    "hull_position": json.dumps(
                        list(
                            box_env.unwrapped.lander.position
                            if engine == "box2d"
                            else body_origin_world(hull)
                        )
                    ),
                    "hull_velocity": json.dumps(list(velocity(hull))),
                    "hull_angle": angle(hull),
                    "hull_angular_velocity": angular_velocity(hull),
                    "left_leg_pose": json.dumps(
                        [*position(bodies[1]), angle(bodies[1])]
                    ),
                    "right_leg_pose": json.dumps(
                        [*position(bodies[2]), angle(bodies[2])]
                    ),
                    "center_of_mass": json.dumps(center.tolist()),
                    "articulated_inertia": inertia,
                    "angular_momentum": momentum,
                    "total_mass": mass,
                }
            )
    box_env.close()
    pymunk_env.close()
    return rows


def response_rows(seed: int, steps: int):
    """Collect matched fixed-action angular-response rows."""
    rows = []
    for sequence_name, make_actions in SEQUENCES.items():
        actions = make_actions(steps)
        box_env = gym.make("LunarLander-v3", disable_env_checker=True)
        box_env.reset(seed=seed)
        box = box_env.unwrapped
        recorder = RecordingRng(box.np_random)
        box.np_random = recorder
        pymunk_env = ExperimentalPymunkLunarLanderEnv()
        pymunk_env.reset(seed=seed)
        for step, action in enumerate(actions, 1):
            for engine in ("box2d", "pymunk"):
                if engine == "box2d":
                    hull, legs = box.lander, box.legs
                    pre_v = np.array(hull.linearVelocity, dtype=float)
                    pre_w, pre_angle, pre_pos = (
                        float(hull.angularVelocity),
                        float(hull.angle),
                        np.array(hull.position, dtype=float),
                    )
                    sample_start = len(recorder.uniform_values)
                    box_env.step(action)
                    dispersion = (
                        np.array(
                            recorder.uniform_values[sample_start : sample_start + 2]
                        )
                        / box_module.SCALE
                    )
                    offset, impulse, point = box_impulse(
                        action, pre_angle, pre_pos, dispersion
                    )
                    dv = np.array(hull.linearVelocity, dtype=float) - pre_v
                    dw = float(hull.angularVelocity) - pre_w
                    theoretical_dv = impulse / float(hull.mass)
                    theoretical_dw = float(
                        (offset[0] * impulse[1] - offset[1] * impulse[0]) / hull.inertia
                    )
                    constraint = [
                        {
                            "motor_impulse": leg.joint.GetMotorTorque(box_module.FPS)
                            / box_module.FPS,
                            "reaction_impulse": leg.joint.GetReactionTorque(
                                box_module.FPS
                            )
                            / box_module.FPS,
                        }
                        for leg in legs
                    ]
                    bodies = [hull, *legs]
                    center, inertia, momentum, mass = articulated_metrics(
                        bodies,
                        lambda body: body.position,
                        lambda body: body.linearVelocity,
                        lambda body: body.angle,
                        lambda body: body.angularVelocity,
                    )
                    leg_data = [
                        [float(leg.angle), float(leg.angularVelocity)] for leg in legs
                    ]
                    hull_v = list(hull.linearVelocity)
                else:
                    demo = pymunk_env.demo
                    pre_v = np.array(demo.lander_body.velocity, dtype=float)
                    pre_w = float(demo.lander_body.angular_velocity)
                    pymunk_env.step(action)
                    telemetry = demo.last_engine_diagnostics or {}
                    impulse = np.array(telemetry.get("impulse", (0.0, 0.0)))
                    point = telemetry.get(
                        "application_point", tuple(demo.lander_body.position)
                    )
                    theoretical_dv = telemetry.get(
                        "theoretical_delta_velocity", (0.0, 0.0)
                    )
                    theoretical_dw = telemetry.get(
                        "theoretical_delta_angular_velocity", 0.0
                    )
                    dv = np.array(demo.lander_body.velocity) - pre_v
                    dw = float(demo.lander_body.angular_velocity) - pre_w
                    diagnostics = demo.physics_diagnostics(action)
                    constraint = [
                        {
                            "motor_impulse": diagnostics[f"{side}_motor_impulse"],
                            "rotary_limit_impulse": diagnostics[
                                f"{side}_rotary_limit_impulse"
                            ],
                        }
                        for side in ("left", "right")
                    ]
                    hull, legs = (
                        demo.lander_body,
                        [demo.left_leg_body, demo.right_leg_body],
                    )
                    center, inertia, momentum, mass = articulated_metrics(
                        [hull, *legs],
                        body_center_of_mass_world,
                        lambda body: body.velocity,
                        lambda body: body.angle,
                        lambda body: body.angular_velocity,
                    )
                    leg_data = [
                        [float(leg.angle), float(leg.angular_velocity)] for leg in legs
                    ]
                    hull_v = list(hull.velocity)
                rows.append(
                    {
                        "engine": engine,
                        "sequence": sequence_name,
                        "seed": seed,
                        "step": step,
                        "action": action,
                        "hull_angle": float(hull.angle),
                        "hull_angular_velocity": float(
                            hull.angularVelocity
                            if engine == "box2d"
                            else hull.angular_velocity
                        ),
                        "hull_velocity": json.dumps(hull_v),
                        "leg_states": json.dumps(leg_data),
                        "impulse_magnitude": float(np.linalg.norm(impulse)),
                        "application_point": json.dumps(list(point)),
                        "theoretical_delta_velocity": json.dumps(list(theoretical_dv)),
                        "observed_delta_velocity": json.dumps(dv.tolist()),
                        "theoretical_delta_angular_velocity": theoretical_dw,
                        "observed_delta_angular_velocity": dw,
                        "constraint_impulses": json.dumps(constraint),
                        "articulated_angular_momentum": momentum,
                        "articulated_inertia": inertia,
                        "total_mass": mass,
                    }
                )
        box_env.close()
        pymunk_env.close()
    return rows


def write_rows(path: Path, rows):
    """Write diagnostic rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run reset and angular-response diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-seeds", type=int, default=1000)
    parser.add_argument("--rollout-seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--initial-output",
        type=Path,
        default=Path("benchmark_results/lunar_lander/angular_initial_states.csv"),
    )
    parser.add_argument(
        "--response-output",
        type=Path,
        default=Path("benchmark_results/lunar_lander/angular_fixed_actions.csv"),
    )
    args = parser.parse_args()
    write_rows(args.initial_output, initial_rows(args.matched_seeds))
    write_rows(args.response_output, response_rows(args.rollout_seed, args.steps))
    print(f"Wrote {args.initial_output}")
    print(f"Wrote {args.response_output}")


if __name__ == "__main__":
    main()
