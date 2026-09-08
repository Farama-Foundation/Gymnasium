"""Standalone and diagnostic helpers for the packaged Pymunk LunarLander."""

from __future__ import annotations

import pymunk

from gymnasium.envs.pymunk.lunar_lander import *  # noqa: F403
from gymnasium.envs.pymunk.lunar_lander import LunarLander, PymunkLunarLanderDemo

ExperimentalPymunkLunarLanderEnv = LunarLander


class DiagnosticPymunkLunarLanderDemo(PymunkLunarLanderDemo):
    """Add impulse telemetry to the packaged physics implementation."""

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

    def _engine_impulse_applied(
        self,
        engine: str,
        dispersion: list[float],
        application_point: pymunk.Vec2d,
        impulse: pymunk.Vec2d,
    ) -> None:
        """Record theoretical impulse response for diagnostic comparisons."""
        origin_offset = application_point - body_origin_world(self.lander_body)  # noqa: F405
        center_of_mass_offset = application_point - body_center_of_mass_world(  # noqa: F405
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


class DiagnosticLunarLander(LunarLander):
    """Use telemetry-enabled physics for comparison scripts."""

    physics_class = DiagnosticPymunkLunarLanderDemo


def physics_diagnostics(
    demo: PymunkLunarLanderDemo,
    action: int,
) -> dict[str, float | int | bool]:
    """Return physical body and constraint diagnostics for one step."""
    bodies = {
        "hull": demo.lander_body,
        "left_leg": demo.left_leg_body,
        "right_leg": demo.right_leg_body,
    }
    diagnostics: dict[str, float | int | bool] = {"action": action}
    for name, body in bodies.items():
        diagnostics[f"{name}_linear_speed"] = float(body.velocity.length)
        diagnostics[f"{name}_angular_speed"] = abs(float(body.angular_velocity))
        diagnostics[f"{name}_is_sleeping"] = bool(body.is_sleeping)
    diagnostics.update(
        {
            "left_leg_contact": demo.left_leg_contact,
            "right_leg_contact": demo.right_leg_contact,
            "hull_kinetic_energy": float(demo.lander_body.kinetic_energy),
            "total_kinetic_energy": float(
                sum(body.kinetic_energy for body in bodies.values())
            ),
            "idle_speed_threshold": float(demo.space.idle_speed_threshold),
            "sleep_time_threshold": float(demo.space.sleep_time_threshold),
        }
    )
    for side_name, leg in (
        ("left", demo.left_leg_body),
        ("right", demo.right_leg_body),
    ):
        for constraint in demo.space.constraints:
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


def main() -> None:
    """Create the standalone Pymunk LunarLander demonstration."""
    PymunkLunarLanderDemo(seed=42)  # noqa: F405


if __name__ == "__main__":
    main()
