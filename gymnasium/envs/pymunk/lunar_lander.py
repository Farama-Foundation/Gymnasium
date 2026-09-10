"""LunarLander environment implemented using Pymunk."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import numpy as np

from gymnasium import Env, error, logger, spaces
from gymnasium.utils import EzPickle

try:
    import pymunk
    from pymunk.util import calc_center
except ImportError as e:
    raise error.DependencyNotInstalled(
        'Pymunk is not installed, run `pip install "gymnasium[pymunk]"`'
    ) from e

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
        gravity: float = -10.0,
    ):
        """Create a seeded Pymunk LunarLander demonstration world."""
        self.world_width = VIEWPORT_WIDTH / SCALE
        self.world_height = VIEWPORT_HEIGHT / SCALE
        rng = np.random.default_rng(seed) if rng is None else rng
        self.rng = rng
        self.space = pymunk.Space()
        self.space.gravity = (0.0, gravity)
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

    def fire_main_engine(
        self,
        dispersion: list[float] | None = None,
        power: float = 1.0,
    ) -> None:
        """Apply main-engine impulse using Box2D-style LunarLander math."""
        tip = pymunk.Vec2d(
            math.sin(self.lander_body.angle),
            math.cos(self.lander_body.angle),
        )
        side = pymunk.Vec2d(-tip.y, tip.x)

        if dispersion is None:
            dispersion = [self.rng.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

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
            -ox * MAIN_ENGINE_POWER * power,
            -oy * MAIN_ENGINE_POWER * power,
        )

        self._engine_impulse_applied("main", dispersion, impulse_pos, impulse)
        self.lander_body.apply_impulse_at_world_point(impulse, impulse_pos)

    def fire_orientation_engine(
        self,
        direction: int,
        dispersion: list[float] | None = None,
        power: float = 1.0,
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
            -ox * SIDE_ENGINE_POWER * power,
            -oy * SIDE_ENGINE_POWER * power,
        )

        self._engine_impulse_applied("side", dispersion, impulse_pos, impulse)
        self.lander_body.apply_impulse_at_world_point(impulse, impulse_pos)

    def step(self, action: int) -> DemoState:
        """Advance the prototype by one step.

        Actions:
        0: no action
        1: fire one orientation engine
        2: fire the main engine
        3: fire the opposite orientation engine
        """
        state, _, _ = self._step_with_powers(action, continuous=False)
        return state

    def _step_with_powers(
        self,
        action: int | np.ndarray,
        *,
        continuous: bool,
    ) -> tuple[DemoState, float, float]:
        """Advance one step and return the applied engine throttle values."""
        dispersion = [self.rng.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        main_power = 0.0
        side_power = 0.0

        if continuous:
            assert isinstance(action, np.ndarray)
            main_action, side_action = (float(value) for value in action)
            if main_action > 0.0:
                main_power = float((np.clip(main_action, 0.0, 1.0) + 1.0) * 0.5)
                self.fire_main_engine(dispersion, main_power)
            if abs(side_action) > 0.5:
                direction = int(np.sign(side_action))
                side_power = float(np.clip(abs(side_action), 0.5, 1.0))
                self.fire_orientation_engine(direction, dispersion, side_power)
        else:
            if action == 1:
                side_power = 1.0
                self.fire_orientation_engine(-1, dispersion)
            elif action == 2:
                main_power = 1.0
                self.fire_main_engine(dispersion)
            elif action == 3:
                side_power = 1.0
                self.fire_orientation_engine(1, dispersion)
            elif action != 0:
                raise ValueError("action must be one of 0, 1, 2, or 3")

        self.space.step(DT)
        return self.state(), main_power, side_power

    def _engine_impulse_applied(
        self,
        _engine: str,
        _dispersion: list[float],
        _application_point: pymunk.Vec2d,
        _impulse: pymunk.Vec2d,
    ) -> None:
        """Provide a no-op hook for out-of-package diagnostic tooling."""

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


class LunarLander(Env, EzPickle):
    r"""A Pymunk implementation of the LunarLander task.

    ## Description
    This environment is a classic rocket trajectory optimization problem. The
    objective is to land a spacecraft safely on a landing pad centered at
    coordinates (0, 0). Fuel is unlimited, and landing outside the pad is
    possible.

    `LunarLander-v4` uses Pymunk for its physics simulation. The earlier
    `LunarLander-v3` uses Box2D and remains available for reproducibility. The
    two implementations preserve the same task, spaces, and reward semantics,
    but do not produce step-for-step identical trajectories.

    ## Action Space
    With the default `continuous=False`, the action space is `Discrete(4)`:
    - 0: do nothing
    - 1: fire the left orientation engine
    - 2: fire the main engine
    - 3: fire the right orientation engine

    With `continuous=True`, the action space is
    `Box(-1, 1, (2,), dtype=np.float32)`. The first value controls the main
    engine: values below 0 turn it off, while values from 0 to 1 scale its
    throttle from 50% to 100%. The second value controls the side engines:
    values between -0.5 and 0.5 turn them off, values below -0.5 fire the left
    engine, and values above 0.5 fire the right engine. Side-engine throttle
    scales from 50% to 100% toward either endpoint.

    ## Observation Space
    The observation is an 8-dimensional `Box`. Its elements are the lander's
    normalized horizontal and vertical position, normalized horizontal and
    vertical velocity, angle, normalized angular velocity, and two indicators
    for left- and right-leg ground contact.

    ## Rewards
    At each nonterminal step, the reward is the change in a shaping value that:
    - increases as the lander approaches the landing pad;
    - increases as linear speed decreases;
    - decreases as the absolute tilt angle increases;
    - adds 10 points for each leg in contact with the ground.

    Firing the main engine costs 0.3 points at full power per step, and firing a
    side engine costs 0.03 points at full power per step. Crashing or leaving
    the horizontal viewport gives a terminal reward of -100. A stable landing
    gives a terminal reward of +100. An episode is considered solved at 200
    points.

    ## Starting State
    The lander starts near the top center of the viewport. A seeded random
    impulse is applied at its center of mass. Terrain generation and all other
    reset randomization use the environment's random-number generator.

    ## Episode End
    The episode terminates when the hull contacts the terrain, the lander leaves
    the horizontal viewport, or a two-leg landing becomes stable. Stability is
    detected by Pymunk articulated-body sleep or by maintaining Box2D-derived
    linear and angular rest thresholds for approximately 0.5 seconds.

    Registered `v4` environments are truncated after 1,000 steps by Gymnasium's
    `TimeLimit` wrapper. Directly constructed environments do not impose an
    internal time limit.

    ## Arguments
    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("LunarLander-v4", continuous=False, gravity=-10.0,
    ...                enable_wind=False, wind_power=15.0,
    ...                turbulence_power=1.5, solver_iterations=180)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<LunarLander<LunarLander-v4>>>>>
    ```

    - `render_mode` can be `None`, `"human"`, or `"rgb_array"`. Human mode
      renders interactively at 50 frames per second; RGB-array mode returns a
      `(400, 600, 3)` `uint8` image.
    - `continuous` selects the discrete or continuous action space described
      above.
    - `gravity` sets vertical gravity and must be strictly between -12 and 0.
      Its default is -10.
    - `enable_wind` enables deterministic, seeded horizontal wind and rotational
      turbulence while the lander is airborne. It is disabled by default.
    - `wind_power` controls maximum linear wind strength. Values from 0 to 20
      are recommended; the default is 15.
    - `turbulence_power` controls maximum rotational turbulence. Values from 0
      to 2 are recommended; the default is 1.5.
    - `solver_iterations` sets the number of Pymunk constraint-solver iterations
      per physics step. The calibrated default is 180.

    ## Version History
    - v4: Reimplemented LunarLander with Pymunk. Both discrete
      `LunarLander-v4` and continuous `LunarLanderContinuous-v4` are available
      through the `pymunk` optional dependency.
    - v3: The Box2D implementation remains available through the `box2d`
      optional dependency.

    ## Installation
    Install the dependencies for the `v4` environments with:

    ```bash
    pip install "gymnasium[pymunk]"
    ```

    ## Implementation Notes
    Pymunk and Box2D differ in contact-friction combination, constraint solving,
    and body-sleep behavior. The Pymunk implementation translates material
    values and applies a consecutive-stability fallback so that task and reward
    semantics remain comparable without requiring trajectory identity.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    physics_class = PymunkLunarLanderDemo

    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        solver_iterations: int = 6 * 30,
    ):
        """Create a Pymunk LunarLander environment."""
        EzPickle.__init__(
            self,
            render_mode=render_mode,
            continuous=continuous,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            solver_iterations=solver_iterations,
        )
        assert -12.0 < gravity and gravity < 0.0, (
            f"gravity (current value: {gravity}) must be between -12 and 0"
        )
        if not 0.0 <= wind_power <= 20.0:
            logger.warn(
                f"wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})"
            )
        if not 0.0 <= turbulence_power <= 2.0:
            logger.warn(
                f"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.continuous = continuous
        self.gravity = gravity
        self.enable_wind = enable_wind
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        self.solver_iterations = solver_iterations
        if continuous:
            self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        else:
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

        self.last_action = 0
        self._main_engine_power = 0.0
        self._side_engine_power = 0.0
        self._side_engine_direction = 0
        self.stable_landing_steps = 0
        self._pygame = None
        self._screen = None
        self._surface = None
        self._clock = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the Pymunk environment."""
        super().reset(seed=seed)
        self.demo = self.physics_class(
            rng=self.np_random,
            randomize_initial_state=True,
            solver_iterations=self.solver_iterations,
            gravity=self.gravity,
        )
        self.prev_shaping = None

        if self.enable_wind:
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        self.last_action = 0
        self._main_engine_power = 0.0
        self._side_engine_power = 0.0
        self._side_engine_direction = 0
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
        self._apply_wind()
        if self.continuous:
            self.demo._step_with_powers(np.zeros(2), continuous=True)
        else:
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
        self.prev_shaping = self._calculate_shaping(observation)
        if self.render_mode == "human":
            self.render()
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

    @staticmethod
    def _calculate_shaping(observation: np.ndarray) -> float:
        """Calculate the observation-derived reward shaping value."""
        return float(
            -100
            * np.sqrt(observation[0] * observation[0] + observation[1] * observation[1])
            - 100
            * np.sqrt(observation[2] * observation[2] + observation[3] * observation[3])
            - 100 * abs(observation[4])
            + 10 * observation[6]
            + 10 * observation[7]
        )

    def step(self, action: int | np.ndarray):
        """Step the Pymunk environment."""
        assert self.demo is not None, "You forgot to call reset()"
        if self.continuous:
            action = np.clip(action, -1, 1).astype(np.float64)
            side_engine_direction = int(np.sign(action[1]))
        else:
            assert self.action_space.contains(action), (
                f"{action!r} ({type(action)}) invalid"
            )
            side_engine_direction = -1 if action == 1 else 1 if action == 3 else 0

        self.last_action = action

        self._apply_wind()
        _, main_power, side_power = self.demo._step_with_powers(
            action,
            continuous=self.continuous,
        )
        self._main_engine_power = main_power
        self._side_engine_power = side_power
        if side_power:
            self._side_engine_direction = side_engine_direction
        else:
            self._side_engine_direction = 0
        observation = self._get_observation()

        shaping = self._calculate_shaping(observation)

        reward = 0.0
        if self.prev_shaping is not None:
            reward = float(shaping - self.prev_shaping)
        self.prev_shaping = shaping

        reward -= main_power * 0.30
        reward -= side_power * 0.03

        terminated = False
        truncated = False
        termination_reason = None
        is_success = False

        if self.demo.crashed or abs(float(observation[0])) >= 1.0:
            terminated = True
            reward = -100.0
            termination_reason = "crash" if self.demo.crashed else "viewport_exit"

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

        info = {
            "termination_reason": termination_reason,
            "is_success": is_success,
            "inside_landing_zone": inside_landing_zone,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _apply_wind(self) -> None:
        """Apply Box2D-compatible wind force and turbulence torque."""
        assert self.demo is not None
        if not self.enable_wind or (
            self.demo.left_leg_contact or self.demo.right_leg_contact
        ):
            return

        wind_magnitude = (
            math.tanh(
                math.sin(0.02 * self.wind_idx)
                + math.sin(math.pi * 0.01 * self.wind_idx)
            )
            * self.wind_power
        )
        self.wind_idx += 1
        # Like Box2D's ApplyForceToCenter, Pymunk integrates this force over
        # DT. Applying the same numeric value as an impulse would be 1 / DT
        # times too strong and would bypass the engines' force-unit contract.
        self.demo.lander_body.apply_force_at_world_point(
            (wind_magnitude, 0.0),
            tuple(body_center_of_mass_world(self.demo.lander_body)),
        )

        torque_magnitude = (
            math.tanh(
                math.sin(0.02 * self.torque_idx)
                + math.sin(math.pi * 0.01 * self.torque_idx)
            )
            * self.turbulence_power
        )
        self.torque_idx += 1
        # Body.torque is likewise integrated over DT into angular impulse.
        self.demo.lander_body.torque += torque_magnitude

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
        body = self.demo.lander_body
        origin = body_origin_world(body)
        tip = pymunk.Vec2d(math.sin(body.angle), math.cos(body.angle))
        side = pymunk.Vec2d(-tip.y, tip.x)

        if self._main_engine_power > 0.0:
            application_point = origin + pymunk.Vec2d(
                tip.x * MAIN_ENGINE_OFFSET,
                -tip.y * MAIN_ENGINE_OFFSET,
            )
            points = [
                application_point - tip * (0.25 + 0.25 * self._main_engine_power),
                application_point - side * 0.18,
                application_point + side * 0.18,
            ]
            pygame.draw.polygon(
                surface,
                (255, 120, 20),
                [self._world_to_screen(point) for point in points],
            )
        if self._side_engine_power > 0.0:
            direction = self._side_engine_direction
            ox = side.x * direction * SIDE_ENGINE_AWAY / SCALE
            oy = -side.y * direction * SIDE_ENGINE_AWAY / SCALE
            application_point = origin + pymunk.Vec2d(
                ox - tip.x * 17 / SCALE,
                oy + tip.y * SIDE_ENGINE_HEIGHT / SCALE,
            )
            exhaust = pymunk.Vec2d(ox, oy).normalized()
            points = [
                application_point + exhaust * (0.2 + 0.2 * self._side_engine_power),
                application_point + tip * 0.12,
                application_point - tip * 0.12,
            ]
            pygame.draw.polygon(
                surface,
                (255, 150, 30),
                [self._world_to_screen(point) for point in points],
            )

    def render(self):
        """Render the current state in the configured mode."""
        if self.render_mode is None:
            env_id = self.spec.id if self.spec is not None else "LunarLander-v4"
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{env_id}", render_mode="rgb_array")'
            )
            return None
        assert self.demo is not None, "You forgot to call reset()"

        if self._pygame is None:
            try:
                self._pygame = importlib.import_module("pygame")
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'pygame is not installed, run `pip install "gymnasium[pymunk]"`'
                ) from e

        pygame = self._pygame
        if not pygame.display.get_init():
            pygame.display.init()
            self._screen = None
        if self._screen is None:
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode(
                    (VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
                )
            else:
                self._screen = pygame.Surface((VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
        if self._clock is None:
            self._clock = pygame.time.Clock()
        if self._surface is None:
            self._surface = pygame.Surface((VIEWPORT_WIDTH, VIEWPORT_HEIGHT))

        surface = self._surface
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

        if self.render_mode == "human":
            self._screen.blit(surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(FPS)
            return None

        return np.transpose(pygame.surfarray.array3d(surface), axes=(1, 0, 2))

    def close(self):
        """Close rendering resources. Safe to call multiple times."""
        if self._pygame is not None and self._pygame.display.get_init():
            self._pygame.display.quit()
        self._screen = None
        self._surface = None
        self._clock = None
        self._pygame = None
