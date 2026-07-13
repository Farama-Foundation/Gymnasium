"""Prototype LunarLander physics implementation using Pymunk.

This script keeps the standalone physics demonstration and an unregistered
experimental Gymnasium Env wrapper for controlled draft comparisons. It is a
proof-of-concept for the physics pieces that would be needed to port
LunarLander from Box2D to Pymunk.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import numpy as np
import pymunk
from pymunk.util import calc_center

FPS = 50
DT = 1.0 / FPS

SCALE = 30.0
VIEWPORT_WIDTH = 600
VIEWPORT_HEIGHT = 400

CHUNKS = 11

GROUND_COLLISION_TYPE = 1
LANDER_COLLISION_TYPE = 2
LEFT_LEG_COLLISION_TYPE = 3
RIGHT_LEG_COLLISION_TYPE = 4

GROUND_CATEGORY = 0b0001
LANDER_CATEGORY = 0b0010
LEG_CATEGORY = 0b0100


INITIAL_RANDOM = 1000.0
INITIAL_RANDOM_ANGLE = 0.05
MAX_EPISODE_STEPS = 1000
SLEEP_TIME_THRESHOLD = 0.5
IDLE_SPEED_THRESHOLD = 0.01
# Box2D defaults: 0.01 m/s, 2 degrees/s, and 0.5 seconds at rest.
STABLE_LINEAR_SPEED_THRESHOLD = 0.01
STABLE_ANGULAR_SPEED_THRESHOLD = math.radians(2.0)
STABLE_LANDING_STEPS = round(SLEEP_TIME_THRESHOLD / DT)

MAIN_ENGINE_POWER = 13.0
MAIN_ENGINE_Y_LOCATION = 4
MAIN_ENGINE_OFFSET = MAIN_ENGINE_Y_LOCATION / SCALE
SIDE_ENGINE_POWER = 0.6
SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12

LEG_AWAY = 20 / SCALE
LEG_DOWN = 18 / SCALE
LEG_WIDTH = 4 / SCALE
LEG_HEIGHT = 16 / SCALE

# Box2D combines fixture friction as sqrt(a * b), while Pymunk multiplies the
# two shape values. Keep terrain at Box2D's 0.1 and convert the body materials
# so the effective contacts match Box2D exactly:
# hull: 0.1 * 1.0 = sqrt(0.1 * 0.1) = 0.1
# legs: 0.1 * sqrt(2) = sqrt(0.1 * 0.2) = sqrt(0.02)
TERRAIN_FRICTION = 0.1
HULL_FRICTION = 1.0
LEG_FRICTION = math.sqrt(2.0)

LANDER_POLY = [
    (-14, 17),
    (-17, 0),
    (-17, -10),
    (17, -10),
    (17, 0),
    (14, 17),
]

_gymnasium = importlib.import_module("gymnasium")
spaces = importlib.import_module("gymnasium.spaces")
GymEnv = _gymnasium.Env


@dataclass
class DemoState:
    """Small state object used by the standalone Pymunk demonstration."""

    x: float
    y: float
    velocity_x: float
    velocity_y: float
    angle: float
    angular_velocity: float
    left_leg_contact: bool
    right_leg_contact: bool
    crashed: bool

    def as_array(self) -> np.ndarray:
        """Return the state as a finite numeric array for simple checks."""
        return np.array(
            [
                self.x,
                self.y,
                self.velocity_x,
                self.velocity_y,
                self.angle,
                self.angular_velocity,
                float(self.left_leg_contact),
                float(self.right_leg_contact),
                float(self.crashed),
            ],
            dtype=np.float64,
        )


@dataclass
class Terrain:
    """Terrain points and helipad metadata."""

    chunk_x: np.ndarray
    smooth_y: np.ndarray
    helipad_x1: float
    helipad_x2: float
    helipad_y: float


def create_terrain(
    space: pymunk.Space,
    rng: np.random.Generator,
    world_width: float,
    world_height: float,
) -> Terrain:
    """Create seeded terrain and add static Pymunk segments to the space."""
    helipad_y = world_height / 4.0
    chunk_x = np.array(
        [world_width / (CHUNKS - 1) * i for i in range(CHUNKS)],
        dtype=np.float64,
    )

    height = rng.uniform(0.0, world_height / 2.0, size=(CHUNKS + 1,))
    height[CHUNKS // 2 - 2] = helipad_y
    height[CHUNKS // 2 - 1] = helipad_y
    height[CHUNKS // 2 + 0] = helipad_y
    height[CHUNKS // 2 + 1] = helipad_y
    height[CHUNKS // 2 + 2] = helipad_y

    smooth_y = np.array(
        [0.33 * (height[i - 1] + height[i] + height[i + 1]) for i in range(CHUNKS)],
        dtype=np.float64,
    )
    # print("\n=== PYMUNK LUNAR LANDER TERRAIN DEBUG ===")
    # print("world_width:", world_width)
    # print("world_height:", world_height)
    # print("CHUNKS:", CHUNKS)

    # print("helipad_x1:", helipad_x1_debug)
    # print("helipad_x2:", helipad_x2_debug)
    # print("helipad_y:", helipad_y)

    # print("chunk_x:")
    # for i, x in enumerate(chunk_x):
    #     print(i, x)

    # print("height:")
    # for i, y in enumerate(height):
    #     print(i, y)

    # print("smooth_y:")
    # for i, y in enumerate(smooth_y):
    #     print(i, y)

    # print("terrain segments:")
    # for i in range(CHUNKS - 1):
    #     print(
    #         i,
    #         "from",
    #         (chunk_x[i], smooth_y[i]),
    #         "to",
    #         (chunk_x[i + 1], smooth_y[i + 1]),
    #     )

    # print("=== END PYMUNK TERRAIN DEBUG ===\n")

    terrain_segments = []
    for i in range(CHUNKS - 1):
        segment = pymunk.Segment(
            space.static_body,
            (float(chunk_x[i]), float(smooth_y[i])),
            (float(chunk_x[i + 1]), float(smooth_y[i + 1])),
            radius=0.0,
        )
        segment.friction = TERRAIN_FRICTION
        segment.elasticity = 0.0
        segment.collision_type = GROUND_COLLISION_TYPE
        segment.filter = pymunk.ShapeFilter(
            categories=GROUND_CATEGORY,
            mask=LANDER_CATEGORY | LEG_CATEGORY,
        )
        space.add(segment)
        terrain_segments.append(segment)

    # print("\n=== PYMUNK TERRAIN COLLISION SHAPES ===")
    # for i, shape in enumerate(terrain_segments):
    #     print(
    #         i,
    #         "a:", shape.a,
    #         "b:", shape.b,
    #         "radius:", shape.radius,
    #         "friction:", shape.friction,
    #         "elasticity:", shape.elasticity,
    #         )
    # print("=== END PYMUNK TERRAIN COLLISION SHAPES ===\n")

    return Terrain(
        chunk_x=chunk_x,
        smooth_y=smooth_y,
        helipad_x1=float(chunk_x[CHUNKS // 2 - 1]),
        helipad_x2=float(chunk_x[CHUNKS // 2 + 1]),
        helipad_y=helipad_y,
    )


def _create_lander_body(
    space: pymunk.Space,
    position: tuple[float, float],
) -> pymunk.Body:
    vertices = [(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
    # Match Box2D debug output for the lander body as closely as possible.
    mass = 4.816666603088379
    moment = 0.8333148956298828

    lander_body = pymunk.Body(mass, moment)
    polygon_center = pymunk.Vec2d(*calc_center(vertices))
    if abs(polygon_center.x) < 1e-15:
        polygon_center = pymunk.Vec2d(0.0, polygon_center.y)
    lander_body.center_of_gravity = polygon_center
    lander_body.position = position
    lander_body.angle = 0.0

    lander_shape = pymunk.Poly(lander_body, vertices)
    lander_shape.friction = HULL_FRICTION
    lander_shape.elasticity = 0.0
    lander_shape.filter = pymunk.ShapeFilter(
        categories=LANDER_CATEGORY,
        mask=GROUND_CATEGORY,
    )
    lander_shape.collision_type = LANDER_COLLISION_TYPE

    # print("\n=== PYMUNK LANDER DEBUG ===")
    # print("initial_position:", position)
    # print("initial_x:", position[0])
    # print("initial_y:", position[1])
    # print("body position:", lander_body.position)
    # print("body angle:", lander_body.angle)
    # print("body velocity:", lander_body.velocity)
    # print("body angular_velocity:", lander_body.angular_velocity)
    # print("body mass:", lander_body.mass)
    # print("body moment:", lander_body.moment)

    # print("lander polygon vertices:")
    # for v in lander_shape.get_vertices():
    #     print(v)

    # for shape in lander_body.shapes:
    #     print("shape friction:", lander_shape.friction)
    #     print("shape elasticity:", lander_shape.elasticity)
    #     print("shape collision_type:", lander_shape.collision_type)
    #     print("shape filter:", lander_shape.filter)

    # print("=== END PYMUNK LANDER DEBUG ===\n")
    space.add(lander_body, lander_shape)
    return lander_body


def body_origin_world(body: pymunk.Body) -> pymunk.Vec2d:
    """Return the Box2D-equivalent body-origin position in world coordinates."""
    return body.local_to_world((0.0, 0.0))


def body_center_of_mass_world(body: pymunk.Body) -> pymunk.Vec2d:
    """Return the body's physical center of mass in world coordinates."""
    return body.local_to_world(body.center_of_gravity)


def create_leg(
    space: pymunk.Space,
    lander_body: pymunk.Body,
    side: int,
    collision_type: int,
) -> pymunk.Body:
    """Create one constrained Pymunk leg for the prototype lander."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")

    # Box2D's box arguments are half-extents. With density=1, the full
    # (4/SCALE) x (16/SCALE) fixture has this mass and box moment.
    leg_mass = LEG_WIDTH * LEG_HEIGHT
    leg_moment = pymunk.moment_for_box(leg_mass, (LEG_WIDTH, LEG_HEIGHT))

    leg_body = pymunk.Body(leg_mass, leg_moment)
    # Box2D creates each leg at ``hull_x - side * LEG_AWAY, hull_y`` with an
    # angle of ``side * 0.05``. Its first position-solver pass then makes the
    # two local anchors coincide. Pymunk has no separate position solver, so
    # initialize directly in the equivalent resolved motor-rest pose.
    reference_angle = side * 0.05
    box2d_joint_angle = -side * 0.4
    relative_angle = reference_angle + box2d_joint_angle
    leg_body.angle = lander_body.angle + relative_angle
    leg_anchor = pymunk.Vec2d(side * LEG_AWAY, LEG_DOWN)
    leg_body.position = lander_body.position - leg_anchor.rotated(leg_body.angle)

    leg_shape = pymunk.Poly.create_box(leg_body, (LEG_WIDTH, LEG_HEIGHT))
    leg_shape.friction = LEG_FRICTION
    leg_shape.elasticity = 0.0
    leg_shape.filter = pymunk.ShapeFilter(
        categories=LEG_CATEGORY,
        mask=GROUND_CATEGORY,
    )
    leg_shape.collision_type = collision_type

    space.add(leg_body, leg_shape)

    pivot = pymunk.PivotJoint(
        lander_body,
        leg_body,
        (0.0, 0.0),
        (side * LEG_AWAY, LEG_DOWN),
    )
    if side == -1:
        minimum_angle = 0.4 + reference_angle
        maximum_angle = 0.9 + reference_angle
    else:
        minimum_angle = -0.9 + reference_angle
        maximum_angle = -0.4 + reference_angle

    rotation_limit = pymunk.RotaryLimitJoint(
        lander_body,
        leg_body,
        minimum_angle,
        maximum_angle,
    )

    # Pymunk's motor rate sign is opposite Box2D's ``motorSpeed`` convention.
    motor = pymunk.SimpleMotor(lander_body, leg_body, -0.3 * side)
    motor.max_force = 40.0

    space.add(pivot, rotation_limit, motor)
    return leg_body


class PymunkLunarLanderDemo:
    """Small action-driven Pymunk LunarLander physics demonstration."""

    def __init__(
        self,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        randomize_initial_state: bool = False,
        solver_iterations: int = 6 * 30,
    ):
        """Create a seeded Pymunk LunarLander demonstration world."""
        self.world_width = VIEWPORT_WIDTH / SCALE
        self.world_height = VIEWPORT_HEIGHT / SCALE
        rng = np.random.default_rng(seed) if rng is None else rng
        self.rng = rng
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -10.0)
        # Retained after a matched trajectory sweep: lower values reduce some
        # airborne errors but fail landing invariants or increase total error.
        self.space.iterations = solver_iterations
        # Pymunk damping is velocity retained per second (applied as
        # damping**dt). 1.0 therefore matches Box2D bodies' zero linear and
        # angular damping.
        self.space.damping = 1.0
        self.space.idle_speed_threshold = IDLE_SPEED_THRESHOLD
        self.space.sleep_time_threshold = SLEEP_TIME_THRESHOLD
        self.terrain = create_terrain(
            self.space,
            rng,
            self.world_width,
            self.world_height,
        )
        self.crashed = False
        self.last_engine_diagnostics: dict[str, object] | None = None
        self.leg_contacts = {
            LEFT_LEG_COLLISION_TYPE: 0,
            RIGHT_LEG_COLLISION_TYPE: 0,
        }

        self.lander_body = _create_lander_body(
            self.space,
            (self.world_width / 2.0, self.world_height),
        )
        self.left_leg_body = create_leg(
            self.space,
            self.lander_body,
            side=-1,
            collision_type=LEFT_LEG_COLLISION_TYPE,
        )
        self.right_leg_body = create_leg(
            self.space,
            self.lander_body,
            side=1,
            collision_type=RIGHT_LEG_COLLISION_TYPE,
        )

        if randomize_initial_state:
            self._randomize_initial_state(rng)

        self._add_collision_handlers()

    def _randomize_initial_state(self, rng: np.random.Generator) -> None:
        """Apply Box2D-style initial random force and small angle perturbation."""
        force = pymunk.Vec2d(
            float(rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)),
            float(rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)),
        )

        # Box2D integrates this force before solving the reset frame. Applying
        # the equivalent one-frame impulse before Pymunk's step preserves the
        # same sampled physical impulse and lets constraints react immediately.
        self.lander_body.apply_impulse_at_world_point(
            force * DT,
            tuple(body_center_of_mass_world(self.lander_body)),
        )
        # Box2d starts at angle 0
        # self.lander_body.angle = float(
        # rng.uniform(-INITIAL_RANDOM_ANGLE, INITIAL_RANDOM_ANGLE)
        # )

    @property
    def left_leg_contact(self) -> bool:
        """Whether the left leg is touching terrain."""
        return self.leg_contacts[LEFT_LEG_COLLISION_TYPE] > 0

    @property
    def right_leg_contact(self) -> bool:
        """Whether the right leg is touching terrain."""
        return self.leg_contacts[RIGHT_LEG_COLLISION_TYPE] > 0

    def _add_collision_handlers(self) -> None:
        def begin_lander_contact(
            arbiter: pymunk.Arbiter,
            _collision_space: pymunk.Space,
            _data: dict,
        ) -> bool:
            self.crashed = True

            return True

        def begin_leg_contact(
            _arbiter: pymunk.Arbiter,
            _collision_space: pymunk.Space,
            data: dict,
        ) -> bool:
            collision_type = data["collision_type"]
            self.leg_contacts[collision_type] += 1

            return True

        def separate_leg_contact(
            _arbiter: pymunk.Arbiter,
            _collision_space: pymunk.Space,
            data: dict,
        ) -> None:
            collision_type = data["collision_type"]
            self.leg_contacts[collision_type] = max(
                0,
                self.leg_contacts[collision_type] - 1,
            )

        self.space.on_collision(
            LEFT_LEG_COLLISION_TYPE,
            GROUND_COLLISION_TYPE,
            begin=begin_leg_contact,
            separate=separate_leg_contact,
            data={"collision_type": LEFT_LEG_COLLISION_TYPE},
        )
        self.space.on_collision(
            RIGHT_LEG_COLLISION_TYPE,
            GROUND_COLLISION_TYPE,
            begin=begin_leg_contact,
            separate=separate_leg_contact,
            data={"collision_type": RIGHT_LEG_COLLISION_TYPE},
        )
        self.space.on_collision(
            LANDER_COLLISION_TYPE,
            GROUND_COLLISION_TYPE,
            begin=begin_lander_contact,
        )

    def fire_main_engine(self, dispersion: list[float] | None = None) -> None:
        """Apply main-engine impulse using Box2D-style LunarLander math."""
        tip = pymunk.Vec2d(
            math.sin(self.lander_body.angle),
            math.cos(self.lander_body.angle),
        )
        side = pymunk.Vec2d(-tip.y, tip.x)

        if dispersion is None:
            dispersion = [self.rng.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 1.0

        ox = (
            tip.x * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
            + side.x * dispersion[1]
        )
        oy = (
            -tip.y * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
            - side.y * dispersion[1]
        )

        origin = body_origin_world(self.lander_body)
        impulse_pos = origin + pymunk.Vec2d(ox, oy)

        impulse = pymunk.Vec2d(
            -ox * MAIN_ENGINE_POWER * m_power,
            -oy * MAIN_ENGINE_POWER * m_power,
        )

        self._record_engine_impulse("main", dispersion, impulse_pos, impulse)
        self.lander_body.apply_impulse_at_world_point(impulse, impulse_pos)

    def fire_orientation_engine(
        self, direction: int, dispersion: list[float] | None = None
    ) -> None:
        """Apply Box2D-style side-engine impulse.

        direction should be:
            -1 for action 1
            +1 for action 3
        """
        tip = pymunk.Vec2d(
            math.sin(self.lander_body.angle),
            math.cos(self.lander_body.angle),
        )
        side = pymunk.Vec2d(-tip.y, tip.x)
        if dispersion is None:
            dispersion = [self.rng.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        s_power = 1.0

        ox = tip.x * dispersion[0] + side.x * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )
        oy = -tip.y * dispersion[0] - side.y * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )

        origin = body_origin_world(self.lander_body)
        impulse_pos = pymunk.Vec2d(
            origin.x + ox - tip.x * 17 / SCALE,
            origin.y + oy + tip.y * SIDE_ENGINE_HEIGHT / SCALE,
        )

        impulse = pymunk.Vec2d(
            -ox * SIDE_ENGINE_POWER * s_power,
            -oy * SIDE_ENGINE_POWER * s_power,
        )

        # print(
        #     "direction=",
        #     direction,
        #     "angle=",
        #     self.lander_body.angle,
        #     "tip=",
        #     tip,
        #     "side=",
        #     side,
        #     "dispersion=",
        #     dispersion,
        #     "ox=",
        #     ox,
        #     "oy=",
        #     oy,
        #     "impulse_pos=",
        #     impulse_pos,
        #     "impulse=",
        #     impulse,
        #     "before_vel=",
        #     self.lander_body.velocity,
        #     "before_ang_vel=",
        #     self.lander_body.angular_velocity,
        # )

        self._record_engine_impulse("side", dispersion, impulse_pos, impulse)
        self.lander_body.apply_impulse_at_world_point(impulse, impulse_pos)

        # print(
        #     "after_vel=",
        #     self.lander_body.velocity,
        #     "after_ang_vel=",
        #     self.lander_body.angular_velocity,
        # )

    def step(self, action: int) -> DemoState:
        """Advance the prototype by one step.

        Actions:
        0: no action
        1: fire one orientation engine
        2: fire the main engine
        3: fire the opposite orientation engine
        """
        velocity_before = pymunk.Vec2d(*self.lander_body.velocity)
        angular_velocity_before = float(self.lander_body.angular_velocity)
        self.last_engine_diagnostics = None
        # Box2D samples dispersion on every physics step, even when no engine
        # fires. Sampling here keeps matched-seed RNG progression identical.
        dispersion = [self.rng.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        if action == 1:
            self.fire_orientation_engine(-1, dispersion)
        elif action == 2:
            self.fire_main_engine(dispersion)
        elif action == 3:
            self.fire_orientation_engine(1, dispersion)
        elif action != 0:
            raise ValueError("action must be one of 0, 1, 2, or 3")

        self.space.step(DT)
        if self.last_engine_diagnostics is not None:
            observed_delta_velocity = self.lander_body.velocity - velocity_before
            self.last_engine_diagnostics["observed_delta_velocity"] = (
                float(observed_delta_velocity.x),
                float(observed_delta_velocity.y),
            )
            self.last_engine_diagnostics["observed_delta_angular_velocity"] = (
                float(self.lander_body.angular_velocity) - angular_velocity_before
            )
        return self.state()

    def _record_engine_impulse(
        self,
        engine: str,
        dispersion: list[float],
        application_point: pymunk.Vec2d,
        impulse: pymunk.Vec2d,
    ) -> None:
        origin_offset = application_point - body_origin_world(self.lander_body)
        center_of_mass_offset = application_point - body_center_of_mass_world(
            self.lander_body
        )
        self.last_engine_diagnostics = {
            "engine": engine,
            "dispersion": tuple(float(value) for value in dispersion),
            "application_point": (
                float(application_point.x),
                float(application_point.y),
            ),
            "application_offset": (
                float(origin_offset.x),
                float(origin_offset.y),
            ),
            "center_of_mass_lever_arm": (
                float(center_of_mass_offset.x),
                float(center_of_mass_offset.y),
            ),
            "impulse": (float(impulse.x), float(impulse.y)),
            "impulse_magnitude": float(impulse.length),
            "theoretical_delta_velocity": (
                float(impulse.x / self.lander_body.mass),
                float(impulse.y / self.lander_body.mass),
            ),
            "theoretical_delta_angular_velocity": float(
                center_of_mass_offset.cross(impulse) / self.lander_body.moment
            ),
        }

    def state(self) -> DemoState:
        """Return the current prototype state."""
        return DemoState(
            x=float(body_origin_world(self.lander_body).x),
            y=float(body_origin_world(self.lander_body).y),
            velocity_x=float(self.lander_body.velocity.x),
            velocity_y=float(self.lander_body.velocity.y),
            angle=float(self.lander_body.angle),
            angular_velocity=float(self.lander_body.angular_velocity),
            left_leg_contact=self.left_leg_contact,
            right_leg_contact=self.right_leg_contact,
            crashed=self.crashed,
        )

    def physics_diagnostics(self, action: int) -> dict[str, float | int | bool]:
        """Return physical body and constraint diagnostics for one step."""
        bodies = {
            "hull": self.lander_body,
            "left_leg": self.left_leg_body,
            "right_leg": self.right_leg_body,
        }
        diagnostics: dict[str, float | int | bool] = {"action": action}
        for name, body in bodies.items():
            diagnostics[f"{name}_linear_speed"] = float(body.velocity.length)
            diagnostics[f"{name}_angular_speed"] = abs(float(body.angular_velocity))
            diagnostics[f"{name}_is_sleeping"] = bool(body.is_sleeping)

        diagnostics.update(
            {
                "left_leg_contact": self.left_leg_contact,
                "right_leg_contact": self.right_leg_contact,
                "hull_kinetic_energy": float(self.lander_body.kinetic_energy),
                "total_kinetic_energy": float(
                    sum(body.kinetic_energy for body in bodies.values())
                ),
                "idle_speed_threshold": float(self.space.idle_speed_threshold),
                "sleep_time_threshold": float(self.space.sleep_time_threshold),
            }
        )
        for side_name, leg in (
            ("left", self.left_leg_body),
            ("right", self.right_leg_body),
        ):
            for constraint in self.space.constraints:
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

    def articulated_angular_momentum(self) -> float:
        """Return total angular momentum about the articulated center of mass."""
        bodies = (self.lander_body, self.left_leg_body, self.right_leg_body)
        total_mass = sum(body.mass for body in bodies)
        center_of_mass = (
            sum(
                (body_center_of_mass_world(body) * body.mass for body in bodies),
                start=pymunk.Vec2d(0.0, 0.0),
            )
            / total_mass
        )
        return float(
            sum(
                body.moment * body.angular_velocity
                + (body_center_of_mass_world(body) - center_of_mass).cross(
                    body.velocity * body.mass
                )
                for body in bodies
            )
        )


class ExperimentalPymunkLunarLanderEnv(GymEnv):
    """Private experimental Gymnasium wrapper for the Pymunk prototype.

    This environment is only for controlled draft comparisons against the
    existing Box2D LunarLander. It is not registered, does not render, and does
    not claim numerical trajectory parity with Box2D.

    As in Box2D, a landing terminates successfully when the articulated lander
    group enters the physics engine's sleeping state with both legs in contact.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        solver_iterations: int = 6 * 30,
    ):
        """Create the unregistered experimental Pymunk LunarLander env."""
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.solver_iterations = solver_iterations
        self.action_space = spaces.Discrete(4)

        low = np.array(
            [
                -2.5,
                -2.5,
                -10.0,
                -10.0,
                -2 * np.pi,
                -10.0,
                -0.0,
                -0.0,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                2.5,
                2.5,
                10.0,
                10.0,
                2 * np.pi,
                10.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low, high)

        self.demo: PymunkLunarLanderDemo | None = None
        self.prev_shaping: float | None = None

        self.elapsed_steps = 0
        self.last_action = 0
        self.stable_landing_steps = 0
        self._pygame = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the experimental Pymunk env."""
        super().reset(seed=seed)
        self.demo = PymunkLunarLanderDemo(
            rng=self.np_random,
            randomize_initial_state=True,
            solver_iterations=self.solver_iterations,
        )
        self.prev_shaping = None

        self.elapsed_steps = 0
        self.last_action = 0
        self.stable_landing_steps = 0
        # Box2D creates legs at hull_y with non-coincident local anchors, then
        # resolves them during reset's world step. Recreate that reset-only
        # pose so the seeded horizontal impulse couples into angular motion.
        for leg_body, side in (
            (self.demo.left_leg_body, -1),
            (self.demo.right_leg_body, 1),
        ):
            leg_body.position = (
                self.demo.lander_body.position.x - side * LEG_AWAY,
                self.demo.lander_body.position.y,
            )
            leg_body.angle = side * 0.05
            leg_body.velocity = (0.0, 0.0)
            leg_body.angular_velocity = 0.0
        self.demo.step(0)
        # Pymunk generates constraint angular velocity after its orientation
        # integration phase; Box2D's reset step exposes the corresponding
        # orientation change immediately.
        for body in (
            self.demo.lander_body,
            self.demo.left_leg_body,
            self.demo.right_leg_body,
        ):
            body.angle += body.angular_velocity * DT
        for leg_body, side in (
            (self.demo.left_leg_body, -1),
            (self.demo.right_leg_body, 1),
        ):
            leg_body.position = self.demo.lander_body.position - pymunk.Vec2d(
                side * LEG_AWAY, LEG_DOWN
            ).rotated(leg_body.angle)
        observation = self._get_observation()
        return observation, {}

    def _get_observation(self) -> np.ndarray:
        assert self.demo is not None

        pos = body_origin_world(self.demo.lander_body)
        vel = self.demo.lander_body.velocity
        state = [
            (pos.x - VIEWPORT_WIDTH / SCALE / 2) / (VIEWPORT_WIDTH / SCALE / 2),
            (pos.y - (self.demo.terrain.helipad_y + LEG_DOWN))
            / (VIEWPORT_HEIGHT / SCALE / 2),
            vel.x * (VIEWPORT_WIDTH / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_HEIGHT / SCALE / 2) / FPS,
            self.demo.lander_body.angle,
            20.0 * self.demo.lander_body.angular_velocity / FPS,
            1.0 if self.demo.left_leg_contact else 0.0,
            1.0 if self.demo.right_leg_contact else 0.0,
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action: int):
        """Step the private experimental Pymunk env."""
        assert self.demo is not None, "You forgot to call reset()"
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )

        self.last_action = action
        self.elapsed_steps += 1
        prev_raw_vy = float(self.demo.lander_body.velocity.y)
        prev_raw_ang_vel = float(self.demo.lander_body.angular_velocity)
        self.demo.step(action)
        observation = self._get_observation()

        shaping = (
            -100
            * np.sqrt(observation[0] * observation[0] + observation[1] * observation[1])
            - 100
            * np.sqrt(observation[2] * observation[2] + observation[3] * observation[3])
            - 100 * abs(observation[4])
            + 10 * observation[6]
            + 10 * observation[7]
        )

        reward = 0.0
        if self.prev_shaping is not None:
            reward = float(shaping - self.prev_shaping)
        self.prev_shaping = float(shaping)

        reward -= 0.30 if action == 2 else 0.0
        reward -= 0.03 if action in (1, 3) else 0.0

        terminated = False
        truncated = False
        termination_reason = None
        is_success = False

        if self.demo.crashed or abs(float(observation[0])) >= 1.0:
            terminated = True
            reward = -100.0
            termination_reason = "crash" if self.demo.crashed else "viewport_exit"

        if not terminated:
            any_leg_contact = self.demo.left_leg_contact or self.demo.right_leg_contact

            raw_vy = float(self.demo.lander_body.velocity.y)
            raw_ang_vel = float(self.demo.lander_body.angular_velocity)

            hard_leg_impact = any_leg_contact and (
                abs(prev_raw_vy) > 3.0
                or abs(prev_raw_ang_vel) > 2.0
                or abs(raw_vy) > 3.0
                or abs(raw_ang_vel) > 2.0
            )

            if hard_leg_impact:
                terminated = True
                reward = -100.0
                termination_reason = "crash"

        inside_landing_zone = False

        if not terminated and not self.demo.crashed:
            if self.demo.left_leg_contact and self.demo.right_leg_contact:
                group_is_sleeping = all(
                    body.is_sleeping
                    for body in (
                        self.demo.lander_body,
                        self.demo.left_leg_body,
                        self.demo.right_leg_body,
                    )
                )
                stable_long_enough = self._update_stable_landing_counter()
                if group_is_sleeping or stable_long_enough:
                    lander_x = float(self.demo.lander_body.position.x)

                    inside_landing_zone = (
                        self.demo.terrain.helipad_x1
                        <= lander_x
                        <= self.demo.terrain.helipad_x2
                    )

                    terminated = True
                    reward = 100.0
                    termination_reason = "stable_landing"
                    is_success = True
            else:
                self.stable_landing_steps = 0

        if not terminated and self.elapsed_steps >= MAX_EPISODE_STEPS:
            truncated = True
            termination_reason = "time_limit"
            is_success = False

        info = {
            "termination_reason": termination_reason,
            "is_success": is_success,
            "inside_landing_zone": inside_landing_zone,
        }

        return observation, reward, terminated, truncated, info

    def _update_stable_landing_counter(self) -> bool:
        """Track Box2D-tolerance stability when native group sleep stalls."""
        assert self.demo is not None
        bodies = (
            self.demo.lander_body,
            self.demo.left_leg_body,
            self.demo.right_leg_body,
        )
        stable = (
            self.demo.left_leg_contact
            and self.demo.right_leg_contact
            and all(
                body.velocity.length <= STABLE_LINEAR_SPEED_THRESHOLD
                and abs(body.angular_velocity) <= STABLE_ANGULAR_SPEED_THRESHOLD
                for body in bodies
            )
        )
        if stable:
            self.stable_landing_steps += 1
        else:
            self.stable_landing_steps = 0
        return self.stable_landing_steps >= STABLE_LANDING_STEPS

    def _world_to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(point[0] * SCALE),
            int(VIEWPORT_HEIGHT - point[1] * SCALE),
        )

    def _body_poly_points(self, body: pymunk.Body) -> list[tuple[int, int]]:
        shape = next(iter(body.shapes))
        return [
            self._world_to_screen(body.local_to_world(vertex))
            for vertex in shape.get_vertices()
        ]

    def _draw_engine_flames(self, pygame, surface) -> None:
        assert self.demo is not None

        if self.last_action == 2:
            points = [
                self.demo.lander_body.local_to_world((0.0, -MAIN_ENGINE_OFFSET - 0.5)),
                self.demo.lander_body.local_to_world((-0.18, -MAIN_ENGINE_OFFSET)),
                self.demo.lander_body.local_to_world((0.18, -MAIN_ENGINE_OFFSET)),
            ]
            pygame.draw.polygon(
                surface,
                (255, 120, 20),
                [self._world_to_screen(point) for point in points],
            )
        elif self.last_action in (1, 3):
            direction = -1 if self.last_action == 1 else 1
            points = [
                self.demo.lander_body.local_to_world(
                    (direction * SIDE_ENGINE_AWAY, SIDE_ENGINE_HEIGHT)
                ),
                self.demo.lander_body.local_to_world(
                    (
                        direction * SIDE_ENGINE_AWAY + direction * 0.35,
                        SIDE_ENGINE_HEIGHT + 0.16,
                    )
                ),
                self.demo.lander_body.local_to_world(
                    (
                        direction * SIDE_ENGINE_AWAY + direction * 0.35,
                        SIDE_ENGINE_HEIGHT - 0.16,
                    )
                ),
            ]
            pygame.draw.polygon(
                surface,
                (255, 150, 30),
                [self._world_to_screen(point) for point in points],
            )

    def render(self):
        """Render a headless RGB frame for qualitative prototype comparison."""
        if self.render_mode != "rgb_array":
            return None
        assert self.demo is not None, "You forgot to call reset()"

        if self._pygame is None:
            self._pygame = importlib.import_module("pygame")

        pygame = self._pygame
        surface = pygame.Surface((VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
        surface.fill((255, 255, 255))

        terrain_points = [
            self._world_to_screen((float(x), float(y)))
            for x, y in zip(
                self.demo.terrain.chunk_x,
                self.demo.terrain.smooth_y,
                strict=True,
            )
        ]
        pygame.draw.polygon(
            surface,
            (20, 20, 20),
            terrain_points + [(VIEWPORT_WIDTH, VIEWPORT_HEIGHT), (0, VIEWPORT_HEIGHT)],
        )
        pygame.draw.lines(surface, (0, 0, 0), False, terrain_points, width=2)

        helipad_start = self._world_to_screen(
            (self.demo.terrain.helipad_x1, self.demo.terrain.helipad_y)
        )
        helipad_end = self._world_to_screen(
            (self.demo.terrain.helipad_x2, self.demo.terrain.helipad_y)
        )
        pygame.draw.line(surface, (40, 180, 80), helipad_start, helipad_end, width=4)

        self._draw_engine_flames(pygame, surface)

        hull_points = self._body_poly_points(self.demo.lander_body)
        pygame.draw.polygon(surface, (128, 102, 230), hull_points)
        pygame.draw.lines(surface, (77, 77, 128), True, hull_points, width=2)

        for leg_body in [self.demo.left_leg_body, self.demo.right_leg_body]:
            leg_points = self._body_poly_points(leg_body)
            pygame.draw.polygon(surface, (128, 102, 230), leg_points)
            pygame.draw.lines(surface, (77, 77, 128), True, leg_points, width=2)

        return np.transpose(pygame.surfarray.array3d(surface), axes=(1, 0, 2))

    def close(self):
        """Close rendering resources. Safe to call multiple times."""
        self._pygame = None


def main() -> None:
    """Run a short action-driven Pymunk LunarLander physics demonstration."""
    PymunkLunarLanderDemo(seed=42)

    # print("Terrain points:")
    # for x, y in zip(demo.terrain.chunk_x, demo.terrain.smooth_y, strict=True):
    #     print(f"({x:.2f}, {y:.2f})")

    # for step in range(FPS * 5):
    #     action = action_schedule.get(step, 0)
    #     state = demo.step(action)

    #     if step % 10 == 0 or action != 0 or state.crashed:
    #         print(
    #             f"step={step:3d} action={action} "
    #             f"position=({state.x:+.3f}, {state.y:+.3f}) "
    #             f"velocity=({state.velocity_x:+.3f}, {state.velocity_y:+.3f}) "
    #             f"angle={state.angle:+.3f} "
    #             f"angular_velocity={state.angular_velocity:+.3f} "
    #             f"left_contact={state.left_leg_contact} "
    #             f"right_contact={state.right_leg_contact} "
    #             f"crashed={state.crashed}"
    #         )

    #     if state.crashed:
    #         break


if __name__ == "__main__":
    main()
