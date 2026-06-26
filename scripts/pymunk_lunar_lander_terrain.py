"""Prototype LunarLander physics implementation using Pymunk."""

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

MAIN_ENGINE_IMPULSE = 0.35
MAIN_ENGINE_OFFSET = 10 / SCALE

SIDE_ENGINE_IMPULSE = 0.03
SIDE_ENGINE_OFFSET = 12 / SCALE

LANDER_POLY = [
    (-14, 17),
    (-17, 0),
    (-17, -10),
    (17, -10),
    (17, 0),
    (14, 17),
]


def create_terrain(
    space: pymunk.Space,
    world_width: float,
    world_height: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create randomized terrain and add its segments to the physics space."""
    rng = np.random.default_rng(seed)

    helipad_y = world_height / 4.0
    chunk_x = np.linspace(0.0, world_width, CHUNKS)

    # Generate uneven terrain around the helipad height.
    terrain_height = rng.uniform(
        0.0,
        world_height / 2.0,
        size=CHUNKS + 2,
    )

    # Flatten the middle section to create a landing pad.
    center = CHUNKS // 2
    terrain_height[center - 1] = helipad_y
    terrain_height[center] = helipad_y
    terrain_height[center + 1] = helipad_y

    # Smooth neighboring terrain heights.
    smooth_y = np.array(
        [
            0.33 * (terrain_height[i] + terrain_height[i + 1] + terrain_height[i + 2])
            for i in range(CHUNKS)
        ]
    )

    center = CHUNKS // 2
    smooth_y[center - 1] = helipad_y
    smooth_y[center] = helipad_y
    smooth_y[center + 1] = helipad_y

    for i in range(CHUNKS - 1):
        segment = pymunk.Segment(
            space.static_body,
            (float(chunk_x[i]), float(smooth_y[i])),
            (float(chunk_x[i + 1]), float(smooth_y[i + 1])),
            radius=0.05,
        )
        segment.friction = 0.5
        segment.elasticity = 0.0
        segment.collision_type = GROUND_COLLISION_TYPE
        space.add(segment)

    return chunk_x, smooth_y


def create_leg(
    space: pymunk.Space,
    lander_body: pymunk.Body,
    side: int,
    collision_type: int,
) -> pymunk.Body:
    """Create one lander leg.

    side=-1 creates the left leg.
    side=+1 creates the right leg.
    """
    leg_width = 4 / SCALE
    leg_height = 18 / SCALE
    leg_away = 20 / SCALE
    leg_down = 18 / SCALE

    leg_mass = 0.25
    leg_moment = pymunk.moment_for_box(
        leg_mass,
        (leg_width, leg_height),
    )

    leg_body = pymunk.Body(leg_mass, leg_moment)

    leg_body.position = lander_body.local_to_world((side * leg_away, -leg_down))
    leg_body.angle = lander_body.angle

    leg_shape = pymunk.Poly.create_box(
        leg_body,
        (leg_width, leg_height),
    )
    leg_shape.friction = 0.5
    leg_shape.elasticity = 0.0
    leg_shape.filter = pymunk.ShapeFilter(group=1)
    leg_shape.collision_type = collision_type

    space.add(leg_body, leg_shape)

    pivot_point = lander_body.local_to_world((side * leg_away, -leg_down / 2))

    pivot = pymunk.PivotJoint(
        lander_body,
        leg_body,
        pivot_point,
    )

    if side == -1:
        minimum_angle = -0.9
        maximum_angle = 0.5
    else:
        minimum_angle = -0.5
        maximum_angle = 0.9

    rotation_limit = pymunk.RotaryLimitJoint(
        lander_body,
        leg_body,
        minimum_angle,
        maximum_angle,
    )

    motor = pymunk.SimpleMotor(
        lander_body,
        leg_body,
        0.0,
    )
    motor.max_force = 40.0

    space.add(pivot, rotation_limit, motor)

    return leg_body


def fire_main_engine(lander_body: pymunk.Body) -> None:
    """Apply an impulse in the lander's upward direction."""
    local_impulse = pymunk.Vec2d(
        0.0,
        MAIN_ENGINE_IMPULSE,
    )

    world_impulse = local_impulse.rotated(lander_body.angle)

    application_point = lander_body.local_to_world((0.0, -MAIN_ENGINE_OFFSET))

    lander_body.apply_impulse_at_world_point(
        world_impulse,
        application_point,
    )


def fire_orientation_engine(
    lander_body: pymunk.Body,
    rotation_direction: int,
) -> None:
    """Apply an off-center impulse that translates and rotates the lander.

    rotation_direction=+1 produces counterclockwise rotation.
    rotation_direction=-1 produces clockwise rotation.
    """
    if rotation_direction not in (-1, 1):
        raise ValueError("rotation_direction must be -1 or +1")

    local_impulse = (
        rotation_direction * SIDE_ENGINE_IMPULSE,
        0.0,
    )

    local_application_point = (
        0.0,
        -SIDE_ENGINE_OFFSET,
    )

    lander_body.apply_impulse_at_local_point(
        local_impulse,
        local_application_point,
    )


def main() -> None:
    """Run the standalone Pymunk LunarLander physics demonstration."""
    space = pymunk.Space()
    space.gravity = (0.0, -10.0)
    game_over = False

    leg_contacts = {
        LEFT_LEG_COLLISION_TYPE: 0,
        RIGHT_LEG_COLLISION_TYPE: 0,
    }

    def begin_lander_contact(
        arbiter: pymunk.Arbiter,
        collision_space: pymunk.Space,
        data: dict,
    ) -> None:
        nonlocal game_over
        game_over = True

    def begin_leg_contact(
        arbiter: pymunk.Arbiter,
        collision_space: pymunk.Space,
        data: dict,
    ) -> None:
        collision_type = data["collision_type"]
        leg_contacts[collision_type] += 1

    def separate_leg_contact(
        arbiter: pymunk.Arbiter,
        collision_space: pymunk.Space,
        data: dict,
    ) -> None:
        collision_type = data["collision_type"]
        leg_contacts[collision_type] = max(
            0,
            leg_contacts[collision_type] - 1,
        )

    world_width = VIEWPORT_WIDTH / SCALE
    world_height = VIEWPORT_HEIGHT / SCALE

    chunk_x, smooth_y = create_terrain(
        space,
        world_width,
        world_height,
    )

    print("Terrain points:")
    for x, y in zip(chunk_x, smooth_y, strict=True):
        print(f"({x:.2f}, {y:.2f})")

    vertices = [(x / SCALE, y / SCALE) for x, y in LANDER_POLY]

    mass = 1.0
    moment = pymunk.moment_for_poly(mass, vertices)

    lander_body = pymunk.Body(mass, moment)
    lander_body.position = (
        world_width / 2.0,
        world_height - 1.0,
    )

    lander_body.angle = 0.0

    lander_shape = pymunk.Poly(lander_body, vertices)
    lander_shape.friction = 0.5
    lander_shape.elasticity = 0.0
    lander_shape.filter = pymunk.ShapeFilter(group=1)
    lander_shape.collision_type = LANDER_COLLISION_TYPE

    space.add(lander_body, lander_shape)

    left_leg_body = create_leg(
        space,
        lander_body,
        side=-1,
        collision_type=LEFT_LEG_COLLISION_TYPE,
    )

    right_leg_body = create_leg(
        space,
        lander_body,
        side=1,
        collision_type=RIGHT_LEG_COLLISION_TYPE,
    )

    space.on_collision(
        LEFT_LEG_COLLISION_TYPE,
        GROUND_COLLISION_TYPE,
        begin=begin_leg_contact,
        separate=separate_leg_contact,
        data={"collision_type": LEFT_LEG_COLLISION_TYPE},
    )

    space.on_collision(
        RIGHT_LEG_COLLISION_TYPE,
        GROUND_COLLISION_TYPE,
        begin=begin_leg_contact,
        separate=separate_leg_contact,
        data={"collision_type": RIGHT_LEG_COLLISION_TYPE},
    )

    space.on_collision(
        LANDER_COLLISION_TYPE,
        GROUND_COLLISION_TYPE,
        begin=begin_lander_contact,
    )

    for step in range(FPS * 5):
        main_engine_on = False

        if main_engine_on:
            fire_main_engine(lander_body)

        side_engine_direction = 0

        if 20 <= step < 30:
            side_engine_direction = 1
        elif 50 <= step < 60:
            side_engine_direction = -1

        if side_engine_direction != 0:
            fire_orientation_engine(
                lander_body,
                rotation_direction=side_engine_direction,
            )

        space.step(DT)

        left_contact = leg_contacts[LEFT_LEG_COLLISION_TYPE] > 0
        right_contact = leg_contacts[RIGHT_LEG_COLLISION_TYPE] > 0

        if step % 10 == 0:
            print(
                f"step={step:3d} "
                f"position=({lander_body.position.x:+.3f}, "
                f"{lander_body.position.y:+.3f}) "
                f"velocity=({lander_body.velocity.x:+.3f}, "
                f"{lander_body.velocity.y:+.3f}) "
                f"angle={lander_body.angle:+.3f} "
                f"main_engine={main_engine_on} "
                f"angular_velocity={lander_body.angular_velocity:+.3f} "
                f"side_engine={side_engine_direction:+d} "
                f"left_angle={left_leg_body.angle:+.3f} "
                f"right_angle={right_leg_body.angle:+.3f} "
                f"left_contact={left_contact} "
                f"right_contact={right_contact}"
                f" game_over={game_over}"
            )


if __name__ == "__main__":
    main()
