"""Test-local observation helpers for the packaged Pymunk LunarLander."""

from __future__ import annotations

import math

import numpy as np
import pymunk

from gymnasium.envs.pymunk.lunar_lander import (
    _LunarLanderPhysics,
    body_center_of_mass_world,
    body_origin_world,
)


class DiagnosticLunarLanderPhysics(_LunarLanderPhysics):
    """Observe engine impulses without changing physics behavior."""

    last_engine_diagnostics: dict[str, object] | None = None

    def step(self, action: int):
        """Step while recording observed hull velocity changes."""
        velocity_before = pymunk.Vec2d(*self.lander_body.velocity)
        angular_velocity_before = float(self.lander_body.angular_velocity)
        self.last_engine_diagnostics = None
        state = super().step(action)
        if self.last_engine_diagnostics is not None:
            observed_delta_velocity = self.lander_body.velocity - velocity_before
            self.last_engine_diagnostics["observed_delta_velocity"] = tuple(
                observed_delta_velocity
            )
            self.last_engine_diagnostics["observed_delta_angular_velocity"] = (
                float(self.lander_body.angular_velocity) - angular_velocity_before
            )
        return state

    def _on_engine_impulse(
        self,
        engine: str,
        dispersion: list[float],
        application_point: pymunk.Vec2d,
        impulse: pymunk.Vec2d,
    ) -> None:
        """Record theoretical impulse response from the production hook."""
        origin_offset = application_point - body_origin_world(self.lander_body)
        center_of_mass_offset = application_point - body_center_of_mass_world(
            self.lander_body
        )
        self.last_engine_diagnostics = {
            "engine": engine,
            "dispersion": tuple(float(value) for value in dispersion),
            "application_point": tuple(application_point),
            "application_offset": tuple(origin_offset),
            "center_of_mass_lever_arm": tuple(center_of_mass_offset),
            "impulse": tuple(impulse),
            "impulse_magnitude": float(impulse.length),
            "theoretical_delta_velocity": tuple(impulse / self.lander_body.mass),
            "theoretical_delta_angular_velocity": float(
                center_of_mass_offset.cross(impulse) / self.lander_body.moment
            ),
        }


def physics_diagnostics(
    physics: _LunarLanderPhysics,
    action: int,
) -> dict[str, float | int | bool]:
    """Read body and constraint diagnostics without mutating the simulation."""
    bodies = {
        "hull": physics.lander_body,
        "left_leg": physics.left_leg_body,
        "right_leg": physics.right_leg_body,
    }
    diagnostics: dict[str, float | int | bool] = {"action": action}
    for name, body in bodies.items():
        diagnostics[f"{name}_linear_speed"] = float(body.velocity.length)
        diagnostics[f"{name}_angular_speed"] = abs(float(body.angular_velocity))
        diagnostics[f"{name}_is_sleeping"] = bool(body.is_sleeping)
    diagnostics.update(
        {
            "left_leg_contact": physics.left_leg_contact,
            "right_leg_contact": physics.right_leg_contact,
            "hull_kinetic_energy": float(physics.lander_body.kinetic_energy),
            "total_kinetic_energy": float(
                sum(body.kinetic_energy for body in bodies.values())
            ),
            "idle_speed_threshold": float(physics.space.idle_speed_threshold),
            "sleep_time_threshold": float(physics.space.sleep_time_threshold),
        }
    )
    for side_name, leg in (
        ("left", physics.left_leg_body),
        ("right", physics.right_leg_body),
    ):
        for constraint in physics.space.constraints:
            if constraint.b is not leg:
                continue
            if isinstance(constraint, pymunk.RotaryLimitJoint):
                diagnostics[f"{side_name}_rotary_limit_impulse"] = abs(
                    float(constraint.impulse)
                )
            elif isinstance(constraint, pymunk.SimpleMotor):
                diagnostics[f"{side_name}_motor_impulse"] = abs(
                    float(constraint.impulse)
                )
    return diagnostics


def box_impulse(action, angle, position, dispersion):
    """Reconstruct a Box2D engine impulse for cross-engine tests."""
    from gymnasium.envs.box2d import lunar_lander as box_module

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


def diagnostic_physics(*args, **kwargs) -> DiagnosticLunarLanderPhysics:
    """Construct test-local telemetry-enabled physics."""
    return DiagnosticLunarLanderPhysics(*args, **kwargs)
