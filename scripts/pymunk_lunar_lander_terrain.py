"""Prototype LunarLander physics implementation using Pymunk.

This script keeps the standalone physics demonstration and an unregistered
experimental Gymnasium Env wrapper for controlled draft comparisons. It is a
proof-of-concept for the physics pieces that would be needed to port
LunarLander from Box2D to Pymunk.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
import pymunk

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

# Draft Pymunk values selected to exercise the prototype mechanics. They are
# not calibrated Box2D equivalents.
INITIAL_RANDOM_IMPULSE = 2.0
INITIAL_RANDOM_ANGLE = 0.05
MAX_EPISODE_STEPS = 1000

MAIN_ENGINE_IMPULSE = 0.35
MAIN_ENGINE_OFFSET = 4 / SCALE

SIDE_ENGINE_IMPULSE = 0.03
SIDE_ENGINE_AWAY = 12 / SCALE
SIDE_ENGINE_HEIGHT = 14 / SCALE

LEG_AWAY = 20 / SCALE
LEG_DOWN = 18 / SCALE
LEG_WIDTH = 4 / SCALE
LEG_HEIGHT = 16 / SCALE

STABLE_LANDING_STEPS = 25
STABLE_VELOCITY_LIMIT = 0.12
STABLE_ANGLE_LIMIT = 0.2
STABLE_ANGULAR_VELOCITY_LIMIT = 0.12

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

    for i in range(CHUNKS - 1):
        segment = pymunk.Segment(
            space.static_body,
            (float(chunk_x[i]), float(smooth_y[i])),
            (float(chunk_x[i + 1]), float(smooth_y[i + 1])),
            radius=0.02,
        )
        segment.friction = 0.1
        segment.elasticity = 0.0
        segment.collision_type = GROUND_COLLISION_TYPE
        segment.filter = pymunk.ShapeFilter(categories=GROUND_CATEGORY)
        space.add(segment)

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
    mass = 1.0
    moment = pymunk.moment_for_poly(mass, vertices)

    lander_body = pymunk.Body(mass, moment)
    lander_body.position = position
    lander_body.angle = 0.0

    lander_shape = pymunk.Poly(lander_body, vertices)
    lander_shape.friction = 0.1
    lander_shape.elasticity = 0.0
    lander_shape.filter = pymunk.ShapeFilter(
        categories=LANDER_CATEGORY,
        mask=GROUND_CATEGORY,
    )
    lander_shape.collision_type = LANDER_COLLISION_TYPE

    space.add(lander_body, lander_shape)
    return lander_body


def create_leg(
    space: pymunk.Space,
    lander_body: pymunk.Body,
    side: int,
    collision_type: int,
) -> pymunk.Body:
    """Create one constrained Pymunk leg for the prototype lander."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")

    leg_mass = 0.25
    leg_moment = pymunk.moment_for_box(leg_mass, (LEG_WIDTH, LEG_HEIGHT))

    leg_body = pymunk.Body(leg_mass, leg_moment)
    leg_body.position = lander_body.local_to_world(
        (side * LEG_AWAY, -LEG_DOWN - LEG_HEIGHT / 2)
    )
    leg_body.angle = lander_body.angle + side * 0.05

    leg_shape = pymunk.Poly.create_box(leg_body, (LEG_WIDTH, LEG_HEIGHT))
    leg_shape.friction = 0.1
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
        (side * LEG_AWAY, -LEG_DOWN),
        (0.0, LEG_HEIGHT / 2),
    )

    if side == -1:
        minimum_angle = -0.9
        maximum_angle = 0.4
        motor_speed = -0.3
    else:
        minimum_angle = -0.4
        maximum_angle = 0.9
        motor_speed = 0.3

    rotation_limit = pymunk.RotaryLimitJoint(
        lander_body,
        leg_body,
        minimum_angle,
        maximum_angle,
    )

    motor = pymunk.SimpleMotor(lander_body, leg_body, motor_speed)
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
    ):
        """Create a seeded Pymunk LunarLander demonstration world."""
        self.world_width = VIEWPORT_WIDTH / SCALE
        self.world_height = VIEWPORT_HEIGHT / SCALE
        rng = np.random.default_rng(seed) if rng is None else rng
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -10.0)
        self.terrain = create_terrain(
            self.space,
            rng,
            self.world_width,
            self.world_height,
        )
        self.crashed = False
        self.leg_contacts = {
            LEFT_LEG_COLLISION_TYPE: 0,
            RIGHT_LEG_COLLISION_TYPE: 0,
        }

        self.lander_body = _create_lander_body(
            self.space,
            (self.world_width / 2.0, self.world_height - 1.0),
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
        """Apply seeded draft reset variation to avoid identical rollouts."""
        self.lander_body.apply_impulse_at_world_point(
            (
                float(rng.uniform(-INITIAL_RANDOM_IMPULSE, INITIAL_RANDOM_IMPULSE)),
                float(rng.uniform(-INITIAL_RANDOM_IMPULSE, INITIAL_RANDOM_IMPULSE)),
            ),
            self.lander_body.position,
        )
        self.lander_body.angle = float(
            rng.uniform(-INITIAL_RANDOM_ANGLE, INITIAL_RANDOM_ANGLE)
        )

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
            _arbiter: pymunk.Arbiter,
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

    def fire_main_engine(self) -> None:
        """Apply an upward main-engine impulse in the lander's local frame."""
        local_impulse = pymunk.Vec2d(0.0, MAIN_ENGINE_IMPULSE)
        impulse = local_impulse.rotated(self.lander_body.angle)
        point = self.lander_body.local_to_world((0.0, -MAIN_ENGINE_OFFSET))
        self.lander_body.apply_impulse_at_world_point(impulse, point)

    def fire_orientation_engine(self, direction: int) -> None:
        """Apply one side-engine impulse.

        direction=-1 and direction=+1 represent opposite orientation engines.
        """
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")

        point = self.lander_body.local_to_world(
            (direction * SIDE_ENGINE_AWAY, SIDE_ENGINE_HEIGHT)
        )
        impulse = pymunk.Vec2d(
            -direction * SIDE_ENGINE_IMPULSE,
            0.0,
        ).rotated(self.lander_body.angle)
        self.lander_body.apply_impulse_at_world_point(impulse, point)

    def step(self, action: int) -> DemoState:
        """Advance the prototype by one step.

        Actions:
        0: no action
        1: fire one orientation engine
        2: fire the main engine
        3: fire the opposite orientation engine
        """
        if action == 1:
            self.fire_orientation_engine(-1)
        elif action == 2:
            self.fire_main_engine()
        elif action == 3:
            self.fire_orientation_engine(1)
        elif action != 0:
            raise ValueError("action must be one of 0, 1, 2, or 3")

        self.space.step(DT)
        return self.state()

    def state(self) -> DemoState:
        """Return the current prototype state."""
        return DemoState(
            x=float(self.lander_body.position.x),
            y=float(self.lander_body.position.y),
            velocity_x=float(self.lander_body.velocity.x),
            velocity_y=float(self.lander_body.velocity.y),
            angle=float(self.lander_body.angle),
            angular_velocity=float(self.lander_body.angular_velocity),
            left_leg_contact=self.left_leg_contact,
            right_leg_contact=self.right_leg_contact,
            crashed=self.crashed,
        )


class ExperimentalPymunkLunarLanderEnv(GymEnv):
    """Private experimental Gymnasium wrapper for the Pymunk prototype.

    This environment is only for controlled draft comparisons against the
    existing Box2D LunarLander. It is not registered, does not render, and does
    not claim numerical trajectory parity with Box2D.

    Temporary stable-landing termination: because this prototype does not map
    Box2D's sleeping behavior exactly, an episode is treated as a successful
    landing after both legs remain in contact while normalized linear velocity,
    absolute angle, and normalized angular velocity stay below small thresholds
    for ``STABLE_LANDING_STEPS`` consecutive simulation steps.
    """

    metadata = {"render_modes": [], "render_fps": FPS}

    def __init__(self):
        """Create the unregistered experimental Pymunk LunarLander env."""
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
        self.stable_landing_steps = 0
        self.elapsed_steps = 0

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
        )
        self.prev_shaping = None
        self.stable_landing_steps = 0
        self.elapsed_steps = 0
        observation, _, _, _, _ = self.step(0)
        return observation, {}

    def _get_observation(self) -> np.ndarray:
        assert self.demo is not None

        pos = self.demo.lander_body.position
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

    def _update_stable_landing_steps(self, observation: np.ndarray) -> None:
        assert self.demo is not None

        stable = (
            self.demo.left_leg_contact
            and self.demo.right_leg_contact
            and not self.demo.crashed
            and abs(float(observation[2])) < STABLE_VELOCITY_LIMIT
            and abs(float(observation[3])) < STABLE_VELOCITY_LIMIT
            and abs(float(observation[4])) < STABLE_ANGLE_LIMIT
            and abs(float(observation[5])) < STABLE_ANGULAR_VELOCITY_LIMIT
        )
        if stable:
            self.stable_landing_steps += 1
        else:
            self.stable_landing_steps = 0

    def step(self, action: int):
        """Step the private experimental Pymunk env."""
        assert self.demo is not None, "You forgot to call reset()"
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )

        self.elapsed_steps += 1
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

        self._update_stable_landing_steps(observation)
        if not terminated and self.stable_landing_steps >= STABLE_LANDING_STEPS:
            terminated = True
            reward = 100.0
            termination_reason = "stable_landing"
            is_success = True

        if not terminated and self.elapsed_steps >= MAX_EPISODE_STEPS:
            truncated = True
            termination_reason = "time_limit"

        info = {
            "termination_reason": termination_reason,
            "is_success": is_success,
        }

        return observation, reward, terminated, truncated, info


def main() -> None:
    """Run a short action-driven Pymunk LunarLander physics demonstration."""
    demo = PymunkLunarLanderDemo(seed=42)

    action_schedule = {
        20: 1,
        21: 1,
        50: 3,
        51: 3,
        100: 2,
        101: 2,
        102: 2,
    }

    print("Terrain points:")
    for x, y in zip(demo.terrain.chunk_x, demo.terrain.smooth_y, strict=True):
        print(f"({x:.2f}, {y:.2f})")

    for step in range(FPS * 5):
        action = action_schedule.get(step, 0)
        state = demo.step(action)

        if step % 10 == 0 or action != 0 or state.crashed:
            print(
                f"step={step:3d} action={action} "
                f"position=({state.x:+.3f}, {state.y:+.3f}) "
                f"velocity=({state.velocity_x:+.3f}, {state.velocity_y:+.3f}) "
                f"angle={state.angle:+.3f} "
                f"angular_velocity={state.angular_velocity:+.3f} "
                f"left_contact={state.left_leg_contact} "
                f"right_contact={state.right_leg_contact} "
                f"crashed={state.crashed}"
            )

        if state.crashed:
            break


if __name__ == "__main__":
    main()
