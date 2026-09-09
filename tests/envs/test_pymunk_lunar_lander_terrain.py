import json
import math
import subprocess
import sys

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

pymunk = pytest.importorskip("pymunk")
try:
    import Box2D
except ImportError:
    Box2D = None

from gymnasium.envs.pymunk.lunar_lander import (  # noqa: E402
    CHUNKS,
    HULL_FRICTION,
    IDLE_SPEED_THRESHOLD,
    LANDER_POLY,
    LEFT_LEG_COLLISION_TYPE,
    LEG_AWAY,
    LEG_DOWN,
    LEG_FRICTION,
    LEG_HEIGHT,
    LEG_WIDTH,
    RIGHT_LEG_COLLISION_TYPE,
    SCALE,
    SLEEP_TIME_THRESHOLD,
    STABLE_ANGULAR_SPEED_THRESHOLD,
    STABLE_LANDING_STEPS,
    STABLE_LINEAR_SPEED_THRESHOLD,
    TERRAIN_FRICTION,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    PymunkLunarLanderDemo,
    body_center_of_mass_world,
    body_origin_world,
)
from gymnasium.envs.pymunk.lunar_lander import (  # noqa: E402
    LunarLander as ExperimentalPymunkLunarLanderEnv,
)

requires_box2d = pytest.mark.skipif(Box2D is None, reason="Box2D is not installed")


def diagnostic_demo(*args, **kwargs):
    """Construct the script-only telemetry-enabled physics helper."""
    from scripts.pymunk_lunar_lander_terrain import DiagnosticPymunkLunarLanderDemo

    return DiagnosticPymunkLunarLanderDemo(*args, **kwargs)


def physics_diagnostics(demo, action):
    """Load physical telemetry only for tests that explicitly require it."""
    from scripts.pymunk_lunar_lander_terrain import physics_diagnostics as collect

    return collect(demo, action)


def test_pymunk_lunar_lander_package_import():
    from gymnasium.envs.pymunk import LunarLander as PackagedLunarLander

    assert PackagedLunarLander is ExperimentalPymunkLunarLanderEnv


def test_pymunk_package_reports_missing_optional_dependency():
    code = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "pymunk" or name.startswith("pymunk."):
        raise ImportError("blocked for test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import gymnasium
from gymnasium.error import DependencyNotInstalled
try:
    import gymnasium.envs.pymunk
except DependencyNotInstalled:
    pass
else:
    raise AssertionError("missing Pymunk did not raise DependencyNotInstalled")
"""

    subprocess.run([sys.executable, "-c", code], check=True)


def test_pymunk_lunar_lander_passes_environment_checker():
    env = ExperimentalPymunkLunarLanderEnv()

    check_env(env)
    env.close()


def place_in_resting_pose(demo):
    """Place the articulated lander just above the flat helipad."""
    relative_angle = 0.4 - 0.05
    left_anchor = pymunk.Vec2d(-LEG_AWAY, LEG_DOWN).rotated(relative_angle)
    left_foot = pymunk.Vec2d(0.0, -LEG_HEIGHT / 2).rotated(relative_angle)
    ground_y = float(demo.terrain.smooth_y[CHUNKS // 2])
    hull_y = ground_y + left_anchor.y - left_foot.y + 0.01

    demo.lander_body.position = (demo.world_width / 2, hull_y)
    demo.lander_body.angle = 0.0
    demo.lander_body.velocity = (0.0, 0.0)
    demo.lander_body.angular_velocity = 0.0
    for body, side, angle in [
        (demo.left_leg_body, -1, relative_angle),
        (demo.right_leg_body, 1, -relative_angle),
    ]:
        body.angle = angle
        body.position = demo.lander_body.position - pymunk.Vec2d(
            side * LEG_AWAY, LEG_DOWN
        ).rotated(angle)
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0


def test_geometry_matches_scaled_box2d_definitions():
    demo = PymunkLunarLanderDemo(seed=123)
    hull_center = np.array([VIEWPORT_WIDTH / SCALE / 2, VIEWPORT_HEIGHT / SCALE])

    assert np.allclose(demo.lander_body.position, hull_center)
    assert min(y for _, y in LANDER_POLY) / SCALE == pytest.approx(-10 / SCALE)

    pivots = {
        constraint.b: constraint
        for constraint in demo.space.constraints
        if isinstance(constraint, pymunk.PivotJoint)
    }
    for leg, side in [
        (demo.left_leg_body, -1),
        (demo.right_leg_body, 1),
    ]:
        reference_angle = side * 0.05
        joint_angle = -side * 0.4
        expected_angle = reference_angle + joint_angle
        local_anchor = pymunk.Vec2d(side * LEG_AWAY, LEG_DOWN)
        expected_center = demo.lander_body.position - local_anchor.rotated(
            expected_angle
        )
        expected_foot_endpoints = [
            expected_center + pymunk.Vec2d(x, -LEG_HEIGHT / 2).rotated(expected_angle)
            for x in (-LEG_WIDTH / 2, LEG_WIDTH / 2)
        ]
        actual_foot_endpoints = [
            leg.local_to_world((x, -LEG_HEIGHT / 2))
            for x in (-LEG_WIDTH / 2, LEG_WIDTH / 2)
        ]
        pivot = pivots[leg]

        assert leg.angle == pytest.approx(expected_angle)
        assert np.allclose(leg.position, expected_center)
        assert np.allclose(demo.lander_body.local_to_world(pivot.anchor_a), hull_center)
        assert np.allclose(leg.local_to_world(pivot.anchor_b), hull_center)
        assert np.allclose(actual_foot_endpoints, expected_foot_endpoints)


@requires_box2d
def test_hull_local_center_origin_and_center_of_mass_match_box2d():
    box_env = gym.make("LunarLander-v3", disable_env_checker=True)
    box_env.reset(seed=123)
    demo = PymunkLunarLanderDemo(seed=123)
    box_hull = box_env.unwrapped.lander
    pymunk_hull = demo.lander_body

    assert tuple(pymunk_hull.center_of_gravity) == pytest.approx(
        tuple(box_hull.localCenter), abs=1e-8
    )
    assert tuple(body_origin_world(pymunk_hull)) == pytest.approx(
        tuple(pymunk_hull.position)
    )
    expected_com = body_origin_world(
        pymunk_hull
    ) + pymunk_hull.center_of_gravity.rotated(pymunk_hull.angle)
    assert tuple(body_center_of_mass_world(pymunk_hull)) == pytest.approx(
        tuple(expected_com)
    )
    assert pymunk_hull.mass == pytest.approx(box_hull.mass)
    assert pymunk_hull.moment == pytest.approx(box_hull.inertia)
    box_env.close()


def test_hull_world_vertices_are_unchanged_by_center_of_gravity():
    corrected = PymunkLunarLanderDemo(seed=123).lander_body
    corrected.position = (10.0, 7.0)
    corrected.angle = 0.2
    baseline = pymunk.Body(corrected.mass, corrected.moment)
    baseline.position = corrected.position
    baseline.angle = corrected.angle
    baseline_shape = pymunk.Poly(
        baseline, [(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
    )
    corrected_shape = next(iter(corrected.shapes))

    baseline_vertices = [
        baseline.local_to_world(vertex) for vertex in baseline_shape.get_vertices()
    ]
    corrected_vertices = [
        corrected.local_to_world(vertex) for vertex in corrected_shape.get_vertices()
    ]
    assert np.allclose(corrected_vertices, baseline_vertices)


def test_observation_position_uses_body_origin_not_center_of_mass():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    origin = body_origin_world(env.demo.lander_body)
    expected_y = (origin.y - (env.demo.terrain.helipad_y + LEG_DOWN)) / (
        VIEWPORT_HEIGHT / SCALE / 2
    )

    observation = env._get_observation()

    assert observation[0] == pytest.approx((origin.x - 10.0) / 10.0)
    assert observation[1] == pytest.approx(expected_y)
    assert body_center_of_mass_world(env.demo.lander_body).y != pytest.approx(origin.y)


def test_terrain_uses_zero_radius_box2d_edges():
    demo = PymunkLunarLanderDemo(seed=123)
    terrain_segments = [
        shape
        for shape in demo.space.static_body.shapes
        if isinstance(shape, pymunk.Segment)
    ]

    assert terrain_segments
    assert all(segment.radius == 0.0 for segment in terrain_segments)


def test_material_friction_matches_box2d_effective_contacts():
    demo = PymunkLunarLanderDemo(seed=123)
    terrain = next(
        shape
        for shape in demo.space.static_body.shapes
        if isinstance(shape, pymunk.Segment)
    )
    hull = next(iter(demo.lander_body.shapes))
    left_leg = next(iter(demo.left_leg_body.shapes))
    right_leg = next(iter(demo.right_leg_body.shapes))

    assert terrain.friction == TERRAIN_FRICTION == pytest.approx(0.1)
    assert hull.friction == HULL_FRICTION == pytest.approx(1.0)
    assert left_leg.friction == LEG_FRICTION == pytest.approx(np.sqrt(2.0))
    assert right_leg.friction == LEG_FRICTION
    assert terrain.friction * hull.friction == pytest.approx(np.sqrt(0.1 * 0.1))
    assert terrain.friction * left_leg.friction == pytest.approx(np.sqrt(0.1 * 0.2))


@requires_box2d
def test_damping_matches_box2d_no_damping_configuration():
    world = Box2D.b2World(gravity=(0.0, -10.0))
    box_body = world.CreateDynamicBody()
    demo = PymunkLunarLanderDemo(seed=123)

    assert box_body.linearDamping == 0.0
    assert box_body.angularDamping == 0.0
    assert demo.space.damping == 1.0


def simulate_box2d_stalled_motor(
    density: float,
    max_torque: float = 40.0,
    steps: int = 100,
) -> tuple[float, float]:
    """Simulate a torque-limited Box2D revolute motor under gravity."""
    dt = 1.0 / 50.0
    width = 0.2
    height = 2.0

    world = Box2D.b2World(gravity=(0.0, -10.0))

    parent = world.CreateStaticBody(position=(0.0, 0.0))
    leg = world.CreateDynamicBody(position=(0.0, -height / 2))

    # Box2D's box dimensions are half-extents.
    leg.CreatePolygonFixture(
        box=(width / 2, height / 2),
        density=density,
    )

    joint = world.CreateRevoluteJoint(
        bodyA=parent,
        bodyB=leg,
        anchor=(0.0, 0.0),
        enableMotor=True,
        motorSpeed=0.3,
        maxMotorTorque=max_torque,
    )

    for _ in range(steps):
        world.Step(dt, 180, 60)

    return float(joint.angle), float(joint.speed)


def simulate_pymunk_stalled_motor(
    density: float,
    max_force: float = 40.0,
    steps: int = 100,
) -> tuple[float, float]:
    """Simulate a torque-limited Pymunk rotary motor under gravity."""
    dt = 1.0 / 50.0
    width = 0.2
    height = 2.0

    space = pymunk.Space()
    space.gravity = (0.0, -10.0)
    space.iterations = 180

    parent = space.static_body

    # Match the Box2D fixture's geometry, density-derived mass, and box inertia.
    mass = width * height * density
    moment = pymunk.moment_for_box(mass, (width, height))

    leg = pymunk.Body(mass, moment)
    leg.position = (0.0, -height / 2)

    shape = pymunk.Poly.create_box(leg, (width, height))

    pivot = pymunk.PivotJoint(
        parent,
        leg,
        (0.0, 0.0),
        (0.0, height / 2),
    )

    # Pymunk's motor rate sign is opposite Box2D's convention for this setup.
    motor = pymunk.SimpleMotor(parent, leg, -0.3)
    motor.max_force = max_force

    space.add(leg, shape, pivot, motor)

    for _ in range(steps):
        space.step(dt)

    return float(leg.angle), float(leg.angular_velocity)


@pytest.mark.parametrize("density", [20.0, 25.0, 30.0])
@requires_box2d
def test_leg_motor_force_mapping_against_box2d(density):
    """Box2D motor torque maps directly to Pymunk max_force."""
    box_angle, box_speed = simulate_box2d_stalled_motor(density)

    direct_angle, direct_speed = simulate_pymunk_stalled_motor(
        density,
        max_force=40.0,
    )
    dt_scaled_angle, _ = simulate_pymunk_stalled_motor(
        density,
        max_force=40.0 / 50.0,
    )

    direct_angle_error = abs(direct_angle - box_angle)
    dt_scaled_angle_error = abs(dt_scaled_angle - box_angle)

    # The load must be strong enough that the torque budget matters.
    # A freely running motor would simply reach the requested 0.3 rad/s.
    assert abs(box_speed) < 0.29
    assert abs(direct_speed) < 0.29

    # Directly mapping Box2D's maxMotorTorque to Pymunk's max_force
    # should reproduce the loaded joint response closely.
    assert direct_angle == pytest.approx(box_angle, abs=0.02)
    assert direct_speed == pytest.approx(box_speed, abs=0.02)

    # Scaling the torque value by dt is a deliberately wrong alternative
    # and should produce a substantially worse match.
    assert direct_angle_error < dt_scaled_angle_error


def simulate_box2d_flat_ground_slide():
    dt = 1 / 50
    width = height = 0.5
    world = Box2D.b2World(gravity=(0.0, -10.0))
    ground = world.CreateStaticBody()
    ground.CreateEdgeFixture(vertices=[(-20.0, 0.0), (20.0, 0.0)], friction=0.1)
    body = world.CreateDynamicBody(position=(0.0, height / 2 + 0.001))
    body.CreatePolygonFixture(
        box=(width / 2, height / 2),
        density=1.0 / (width * height),
        friction=0.2,
    )
    body.linearVelocity = (1.0, 0.0)
    start_x = float(body.position.x)
    speeds = []
    for _step in range(1, 1001):
        world.Step(dt, 180, 60)
        speeds.append(abs(float(body.linearVelocity.x)))
        if not body.awake:
            break
    return speeds, float(body.position.x) - start_x, _step * dt, not body.awake


def simulate_pymunk_flat_ground_slide():
    dt = 1 / 50
    width = height = 0.5
    space = pymunk.Space()
    space.gravity = (0.0, -10.0)
    space.iterations = 30
    space.idle_speed_threshold = IDLE_SPEED_THRESHOLD
    space.sleep_time_threshold = SLEEP_TIME_THRESHOLD
    ground = pymunk.Segment(space.static_body, (-20.0, 0.0), (20.0, 0.0), 0.0)
    ground.friction = TERRAIN_FRICTION
    body = pymunk.Body(1.0, pymunk.moment_for_box(1.0, (width, height)))
    body.position = (0.0, height / 2 + 0.001)
    shape = pymunk.Poly.create_box(body, (width, height))
    shape.friction = LEG_FRICTION
    space.add(ground, body, shape)
    body.velocity = (1.0, 0.0)
    start_x = float(body.position.x)
    speeds = []
    for _step in range(1, 1001):
        space.step(dt)
        speeds.append(abs(float(body.velocity.x)))
        if body.is_sleeping:
            break
    return speeds, float(body.position.x) - start_x, _step * dt, body.is_sleeping


@requires_box2d
def test_flat_ground_slide_matches_box2d_effective_friction():
    box_speeds, box_distance, box_time, box_sleeping = (
        simulate_box2d_flat_ground_slide()
    )
    pymunk_speeds, pymunk_distance, pymunk_time, pymunk_sleeping = (
        simulate_pymunk_flat_ground_slide()
    )

    assert box_sleeping
    assert pymunk_sleeping
    assert box_speeds[-1] < STABLE_LINEAR_SPEED_THRESHOLD
    assert pymunk_speeds[-1] < STABLE_LINEAR_SPEED_THRESHOLD
    assert pymunk_distance == pytest.approx(box_distance, rel=0.1)
    assert pymunk_time == pytest.approx(box_time, abs=2 / 50)


def test_sleep_configuration_matches_box2d_style_behavior():
    demo = PymunkLunarLanderDemo(seed=123)

    assert demo.space.idle_speed_threshold == IDLE_SPEED_THRESHOLD
    assert demo.space.sleep_time_threshold == SLEEP_TIME_THRESHOLD
    assert SLEEP_TIME_THRESHOLD == pytest.approx(0.5)
    assert STABLE_LINEAR_SPEED_THRESHOLD == pytest.approx(0.01)
    assert STABLE_ANGULAR_SPEED_THRESHOLD == pytest.approx(np.deg2rad(2.0))
    assert STABLE_LANDING_STEPS == 25


def test_solver_iterations_are_configurable_without_changing_default():
    default_env = ExperimentalPymunkLunarLanderEnv()
    diagnostic_env = ExperimentalPymunkLunarLanderEnv(solver_iterations=30)
    default_env.reset(seed=123)
    diagnostic_env.reset(seed=123)

    assert default_env.demo.space.iterations == 180
    assert diagnostic_env.demo.space.iterations == 30


@requires_box2d
def test_selected_solver_improves_trajectory_error_over_30_iterations():
    from scripts.sweep_pymunk_lunar_lander_solver_iterations import (
        aggregate_scores,
        sweep,
    )

    scores = aggregate_scores(sweep([30, 180], range(100, 102), steps=100))

    assert scores[180] < scores[30]


def test_resting_articulated_group_sleeps_after_at_least_one_second():
    demo = PymunkLunarLanderDemo(seed=123)
    place_in_resting_pose(demo)

    for _ in range(100):
        demo.step(0)

    assert demo.lander_body.is_sleeping
    assert demo.left_leg_body.is_sleeping
    assert demo.right_leg_body.is_sleeping


def test_constraint_diagnostics_expose_motor_and_limit_impulses():
    demo = PymunkLunarLanderDemo(seed=123)
    place_in_resting_pose(demo)
    for _ in range(25):
        demo.step(0)
    diagnostics = physics_diagnostics(demo, 0)

    assert diagnostics["left_motor_impulse"] > 0
    assert diagnostics["right_motor_impulse"] > 0
    assert diagnostics["left_rotary_limit_impulse"] > 0
    assert diagnostics["right_rotary_limit_impulse"] > 0
    assert diagnostics["left_motor_impulse"] == pytest.approx(40.0 / 50.0)
    assert diagnostics["right_motor_impulse"] == pytest.approx(40.0 / 50.0)


def test_corrected_leg_friction_reduces_articulated_landing_drift():
    corrected_demo = PymunkLunarLanderDemo(seed=123)
    frictionless_demo = PymunkLunarLanderDemo(seed=123)
    for demo in (corrected_demo, frictionless_demo):
        place_in_resting_pose(demo)
        for body in (
            demo.lander_body,
            demo.left_leg_body,
            demo.right_leg_body,
        ):
            body.velocity = (0.05, 0.0)
    for leg in (frictionless_demo.left_leg_body, frictionless_demo.right_leg_body):
        next(iter(leg.shapes)).friction = 0.0

    corrected_start_x = float(corrected_demo.lander_body.position.x)
    frictionless_start_x = float(frictionless_demo.lander_body.position.x)
    for _ in range(100):
        corrected_demo.step(0)
        frictionless_demo.step(0)

    corrected_drift = abs(corrected_demo.lander_body.position.x - corrected_start_x)
    frictionless_drift = abs(
        frictionless_demo.lander_body.position.x - frictionless_start_x
    )
    corrected_diagnostics = physics_diagnostics(corrected_demo, 0)

    assert corrected_drift < frictionless_drift / 5
    assert corrected_demo.lander_body.is_sleeping
    assert corrected_diagnostics["hull_linear_speed"] < STABLE_LINEAR_SPEED_THRESHOLD
    assert frictionless_demo.lander_body.velocity.length > STABLE_LINEAR_SPEED_THRESHOLD


def test_friction_conversion_does_not_change_airborne_trajectory():
    corrected_demo = PymunkLunarLanderDemo(seed=123)
    legacy_demo = PymunkLunarLanderDemo(seed=123)
    for shape in legacy_demo.space.shapes:
        if shape.body is not legacy_demo.space.static_body:
            shape.friction = 0.0

    actions = [0, 1, 0, 2, 0, 3, 0, 0, 2, 0]
    for action in actions:
        corrected_state = corrected_demo.step(action)
        legacy_state = legacy_demo.step(action)
        assert not corrected_state.left_leg_contact
        assert not corrected_state.right_leg_contact
        assert np.allclose(corrected_state.as_array(), legacy_state.as_array())


def test_seeded_terrain_is_reproducible():
    first_demo = PymunkLunarLanderDemo(seed=123)
    second_demo = PymunkLunarLanderDemo(seed=123)
    different_demo = PymunkLunarLanderDemo(seed=456)

    assert np.array_equal(first_demo.terrain.chunk_x, second_demo.terrain.chunk_x)
    assert np.array_equal(first_demo.terrain.smooth_y, second_demo.terrain.smooth_y)
    assert not np.array_equal(
        first_demo.terrain.smooth_y, different_demo.terrain.smooth_y
    )

    center = CHUNKS // 2
    helipad_points = first_demo.terrain.smooth_y[center - 1 : center + 2]
    assert np.allclose(helipad_points, helipad_points[0])


def test_state_values_are_finite():
    demo = PymunkLunarLanderDemo(seed=123)

    for _ in range(20):
        state = demo.step(0)

    assert np.isfinite(state.as_array()).all()


@requires_box2d
def test_matched_seed_initial_state_distribution_moments_match_box2d():
    from scripts.analyze_lunar_lander_angular_dynamics import initial_rows

    rows = initial_rows(1000)
    box_rows = [row for row in rows if row["engine"] == "box2d"]
    pymunk_rows = [row for row in rows if row["engine"] == "pymunk"]

    box_vx = np.array([json.loads(row["hull_velocity"])[0] for row in box_rows])
    pymunk_vx = np.array([json.loads(row["hull_velocity"])[0] for row in pymunk_rows])
    box_omega = np.array([row["hull_angular_velocity"] for row in box_rows])
    pymunk_omega = np.array([row["hull_angular_velocity"] for row in pymunk_rows])

    box_vy = np.array([json.loads(row["hull_velocity"])[1] for row in box_rows])
    pymunk_vy = np.array([json.loads(row["hull_velocity"])[1] for row in pymunk_rows])
    box_angle = np.array([row["hull_angle"] for row in box_rows])
    pymunk_angle = np.array([row["hull_angle"] for row in pymunk_rows])

    assert np.std(pymunk_vx) == pytest.approx(np.std(box_vx), rel=0.02)
    assert np.std(pymunk_vy) == pytest.approx(np.std(box_vy), rel=0.02)
    assert np.std(pymunk_angle) == pytest.approx(np.std(box_angle), rel=0.25)
    assert np.std(pymunk_omega) == pytest.approx(np.std(box_omega), rel=0.25)
    assert np.mean(pymunk_vx) == pytest.approx(np.mean(box_vx), abs=0.01)
    assert np.mean(pymunk_vy) == pytest.approx(np.mean(box_vy), abs=0.1)
    assert pymunk_rows[0]["total_mass"] == pytest.approx(
        box_rows[0]["total_mass"], rel=1e-6
    )
    assert pymunk_rows[0]["articulated_inertia"] == pytest.approx(
        box_rows[0]["articulated_inertia"], rel=0.01
    )


def test_leg_mass_and_moment_match_box2d():
    demo = PymunkLunarLanderDemo(seed=123)
    expected_mass = (4 / SCALE) * (16 / SCALE)
    expected_moment = expected_mass * ((4 / SCALE) ** 2 + (16 / SCALE) ** 2) / 12

    for leg in (demo.left_leg_body, demo.right_leg_body):
        assert leg.mass == pytest.approx(expected_mass)
        assert leg.moment == pytest.approx(expected_moment)


def remove_pymunk_legs_and_constraints(demo):
    demo.space.remove(*list(demo.space.constraints))
    for leg in (demo.left_leg_body, demo.right_leg_body):
        demo.space.remove(*list(leg.shapes), leg)


@pytest.mark.parametrize("action", [1, 2, 3])
def test_one_body_engine_telemetry_matches_theoretical_impulse(action):
    demo = diagnostic_demo(seed=123)
    remove_pymunk_legs_and_constraints(demo)
    demo.space.gravity = (0.0, 0.0)
    demo.lander_body.velocity = (0.0, 0.0)
    demo.lander_body.angular_velocity = 0.0

    demo.step(action)
    telemetry = demo.last_engine_diagnostics

    assert telemetry is not None
    assert telemetry["observed_delta_velocity"] == pytest.approx(
        telemetry["theoretical_delta_velocity"], abs=1e-12
    )
    assert telemetry["observed_delta_angular_velocity"] == pytest.approx(
        telemetry["theoretical_delta_angular_velocity"], abs=1e-12
    )


def test_one_body_no_action_has_zero_angular_response():
    demo = diagnostic_demo(seed=123)
    remove_pymunk_legs_and_constraints(demo)
    demo.space.gravity = (0.0, 0.0)
    demo.lander_body.velocity = (0.0, 0.0)
    demo.lander_body.angular_velocity = 0.0

    state = demo.step(0)

    assert demo.last_engine_diagnostics is None
    assert state.velocity_x == pytest.approx(0.0)
    assert state.velocity_y == pytest.approx(0.0)
    assert state.angular_velocity == pytest.approx(0.0)


def test_articulated_constraints_change_side_engine_hull_response():
    isolated_demo = diagnostic_demo(seed=123)
    articulated_demo = diagnostic_demo(seed=123)
    remove_pymunk_legs_and_constraints(isolated_demo)
    for demo in (isolated_demo, articulated_demo):
        demo.space.gravity = (0.0, 0.0)
        demo.lander_body.velocity = (0.0, 0.0)
        demo.lander_body.angular_velocity = 0.0
        demo.rng = iter_uniform_rng([0.3, -0.6])
    for leg in (articulated_demo.left_leg_body, articulated_demo.right_leg_body):
        leg.velocity = (0.0, 0.0)
        leg.angular_velocity = 0.0

    isolated_demo.step(3)
    articulated_demo.step(3)
    isolated_delta = isolated_demo.last_engine_diagnostics[
        "observed_delta_angular_velocity"
    ]
    articulated_delta = articulated_demo.last_engine_diagnostics[
        "observed_delta_angular_velocity"
    ]
    diagnostics = physics_diagnostics(articulated_demo, 3)

    assert isolated_delta == pytest.approx(
        isolated_demo.last_engine_diagnostics["theoretical_delta_angular_velocity"]
    )
    assert articulated_delta == pytest.approx(isolated_delta, rel=0.15)
    assert 0 < diagnostics["left_motor_impulse"] <= 0.8
    assert 0 < diagnostics["right_motor_impulse"] <= 0.8


@pytest.mark.parametrize("action", [1, 3])
@requires_box2d
def test_side_engine_impulse_and_application_point_match_box2d(action):
    from scripts.analyze_lunar_lander_angular_dynamics import box_impulse

    angle = 0.2
    position = np.array([10.0, 10.0])
    dispersion = np.array([0.01, -0.02])
    box_offset, box_engine_impulse, _ = box_impulse(action, angle, position, dispersion)
    demo = diagnostic_demo(seed=123)
    demo.lander_body.position = tuple(position)
    demo.lander_body.angle = angle
    demo.rng = iter_uniform_rng(dispersion * SCALE)

    demo.fire_orientation_engine(action - 2)
    telemetry = demo.last_engine_diagnostics
    pymunk_offset = np.array(telemetry["application_offset"])
    pymunk_impulse = np.array(telemetry["impulse"])
    assert pymunk_impulse == pytest.approx(box_engine_impulse)
    assert pymunk_offset == pytest.approx(box_offset)
    expected_lever_arm = box_offset - np.array(
        demo.lander_body.center_of_gravity.rotated(angle)
    )
    assert telemetry["center_of_mass_lever_arm"] == pytest.approx(expected_lever_arm)
    box_torque = (
        expected_lever_arm[0] * box_engine_impulse[1]
        - expected_lever_arm[1] * box_engine_impulse[0]
    )
    pymunk_lever_arm = np.array(telemetry["center_of_mass_lever_arm"])
    pymunk_torque = (
        pymunk_lever_arm[0] * pymunk_impulse[1]
        - pymunk_lever_arm[1] * pymunk_impulse[0]
    )
    assert np.sign(box_torque) == np.sign(pymunk_torque)


def iter_uniform_rng(values):
    class Rng:
        def __init__(self, samples):
            self.samples = iter(samples)

        def uniform(self, low, high):
            return next(self.samples)

    return Rng(values)


@requires_box2d
def test_main_engine_impulse_and_application_point_match_box2d():
    from scripts.analyze_lunar_lander_angular_dynamics import box_impulse

    angle = -0.3
    position = np.array([10.0, 10.0])
    dispersion = np.array([0.015, 0.005])
    box_offset, box_engine_impulse, _ = box_impulse(2, angle, position, dispersion)
    demo = diagnostic_demo(seed=123)
    demo.lander_body.position = tuple(position)
    demo.lander_body.angle = angle
    demo.rng = iter_uniform_rng(dispersion * SCALE)

    demo.fire_main_engine()
    telemetry = demo.last_engine_diagnostics

    assert telemetry["application_offset"] == pytest.approx(box_offset)
    assert telemetry["impulse"] == pytest.approx(box_engine_impulse)
    expected_lever_arm = box_offset - np.array(
        demo.lander_body.center_of_gravity.rotated(angle)
    )
    assert telemetry["center_of_mass_lever_arm"] == pytest.approx(expected_lever_arm)


def test_main_thrust_reduces_downward_velocity():
    passive_demo = PymunkLunarLanderDemo(seed=123)
    thrust_demo = PymunkLunarLanderDemo(seed=123)

    passive_state = passive_demo.step(0)
    thrust_state = thrust_demo.step(2)

    assert thrust_state.velocity_y > passive_state.velocity_y


def test_orientation_actions_produce_opposite_angular_acceleration():
    passive_demo = PymunkLunarLanderDemo(seed=123)
    left_demo = PymunkLunarLanderDemo(seed=123)
    right_demo = PymunkLunarLanderDemo(seed=123)

    passive_state = passive_demo.step(0)
    left_state = left_demo.step(1)
    right_state = right_demo.step(3)

    assert left_state.angular_velocity > passive_state.angular_velocity
    assert right_state.angular_velocity < passive_state.angular_velocity


def test_leg_contacts_become_active_after_normal_landing():
    demo = PymunkLunarLanderDemo(seed=42)
    place_in_resting_pose(demo)

    for _ in range(200):
        state = demo.step(0)
        if demo.lander_body.is_sleeping:
            break

    assert state.left_leg_contact
    assert state.right_leg_contact
    assert not state.crashed


def test_hull_contact_sets_crash_flag():
    demo = PymunkLunarLanderDemo(seed=123)
    demo.lander_body.position = (
        demo.world_width / 2.0,
        demo.terrain.helipad_y + 0.05,
    )
    demo.lander_body.velocity = (0.0, 0.0)
    demo.left_leg_body.position = (0.0, demo.world_height)
    demo.right_leg_body.position = (demo.world_width, demo.world_height)

    state = demo.step(0)

    assert state.crashed


def test_experimental_env_reset_and_step_contracts():
    env = ExperimentalPymunkLunarLanderEnv()

    observation, info = env.reset(seed=123)
    assert observation.shape == (8,)
    assert observation.dtype == np.float32
    assert info == {}

    next_observation, reward, terminated, truncated, info = env.step(0)
    assert next_observation.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert info == {
        "termination_reason": None,
        "is_success": False,
        "inside_landing_zone": False,
    }


def shaping(observation):
    """Calculate the LunarLander shaping formula independently of the environment."""
    return float(
        -100 * np.hypot(observation[0], observation[1])
        - 100 * np.hypot(observation[2], observation[3])
        - 100 * abs(observation[4])
        + 10 * observation[6]
        + 10 * observation[7]
    )


def test_reset_initializes_reward_shaping_from_returned_observation():
    env = ExperimentalPymunkLunarLanderEnv()

    observation, _ = env.reset(seed=123)

    assert env.prev_shaping == pytest.approx(shaping(observation))
    env.close()


@pytest.mark.parametrize(
    ("action", "fuel_penalty"),
    [(0, 0.0), (1, 0.03), (2, 0.30), (3, 0.03)],
)
def test_first_discrete_reward_uses_reset_shaping_and_fuel_penalty(
    action, fuel_penalty
):
    env = ExperimentalPymunkLunarLanderEnv()
    reset_observation, _ = env.reset(seed=123)

    observation, reward, terminated, _, _ = env.step(action)

    assert not terminated
    assert reward == pytest.approx(
        shaping(observation) - shaping(reset_observation) - fuel_penalty
    )
    env.close()


@pytest.mark.parametrize(
    ("action", "fuel_penalty"),
    [
        (np.array([0.0, 0.0]), 0.0),
        (np.array([0.5, 0.0]), 0.75 * 0.30),
        (np.array([0.0, -0.75]), 0.75 * 0.03),
    ],
)
def test_first_continuous_reward_uses_reset_shaping_and_throttle_penalty(
    action, fuel_penalty
):
    env = ExperimentalPymunkLunarLanderEnv(continuous=True)
    reset_observation, _ = env.reset(seed=123)

    observation, reward, terminated, _, _ = env.step(action)

    assert not terminated
    assert reward == pytest.approx(
        shaping(observation) - shaping(reset_observation) - fuel_penalty
    )
    env.close()


def test_repeated_reset_replaces_previous_episode_shaping_baseline():
    env = ExperimentalPymunkLunarLanderEnv()
    first_observation, _ = env.reset(seed=123)
    first_baseline = env.prev_shaping
    env.step(2)

    second_observation, _ = env.reset(seed=456)

    assert first_baseline == pytest.approx(shaping(first_observation))
    assert env.prev_shaping == pytest.approx(shaping(second_observation))
    assert env.prev_shaping != pytest.approx(first_baseline)
    env.close()


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_first_step_reward_is_deterministic_for_same_seed_and_action(action):
    first = ExperimentalPymunkLunarLanderEnv()
    second = ExperimentalPymunkLunarLanderEnv()
    first.reset(seed=123)
    second.reset(seed=123)

    first_step = first.step(action)
    second_step = second.step(action)

    assert np.array_equal(first_step[0], second_step[0])
    assert first_step[1:] == second_step[1:]
    first.close()
    second.close()


def test_first_and_second_user_rewards_decompose_from_reset_shaping():
    env = ExperimentalPymunkLunarLanderEnv()
    reset_observation, _ = env.reset(seed=123)

    first_observation, first_reward, first_terminated, _, _ = env.step(2)
    second_observation, second_reward, second_terminated, _, _ = env.step(1)

    assert not first_terminated
    assert not second_terminated
    assert first_reward == pytest.approx(
        shaping(first_observation) - shaping(reset_observation) - 0.30
    )
    assert second_reward == pytest.approx(
        shaping(second_observation) - shaping(first_observation) - 0.03
    )
    assert first_reward + second_reward == pytest.approx(
        shaping(second_observation) - shaping(reset_observation) - 0.33
    )
    env.close()


@pytest.mark.parametrize(
    ("continuous", "action"),
    [(False, 2), (True, np.array([0.5, -0.75]))],
)
def test_reward_fix_does_not_change_ef1cd8dfa_physics_or_terminal_flags(
    continuous, action
):
    corrected = ExperimentalPymunkLunarLanderEnv(continuous=continuous)
    historical = ExperimentalPymunkLunarLanderEnv(continuous=continuous)
    corrected_observation, _ = corrected.reset(seed=123)
    historical_observation, _ = historical.reset(seed=123)
    # ef1cd8dfa advanced identical reset physics but left this baseline unset.
    historical.prev_shaping = None

    corrected_step = corrected.step(action)
    historical_step = historical.step(action)

    assert np.array_equal(corrected_observation, historical_observation)
    assert np.array_equal(corrected_step[0], historical_step[0])
    assert corrected_step[2:] == historical_step[2:]
    assert corrected_step[1] != pytest.approx(historical_step[1])
    corrected.close()
    historical.close()


def test_first_noop_reward_does_not_repeat_historical_zero_reward_failure():
    env = ExperimentalPymunkLunarLanderEnv()
    reset_observation, _ = env.reset(seed=123)

    observation, reward, terminated, _, _ = env.step(0)

    assert not terminated
    assert reward == pytest.approx(shaping(observation) - shaping(reset_observation))
    assert reward != pytest.approx(0.0)
    env.close()


@requires_box2d
def test_reset_shaping_initialization_matches_box2d_reward_contract():
    box2d = gym.make("LunarLander-v3", disable_env_checker=True).unwrapped
    pymunk_env = ExperimentalPymunkLunarLanderEnv()
    box_observation, _ = box2d.reset(seed=123)
    pymunk_observation, _ = pymunk_env.reset(seed=123)

    assert box2d.prev_shaping == pytest.approx(shaping(box_observation))
    assert pymunk_env.prev_shaping == pytest.approx(shaping(pymunk_observation))
    box2d.close()
    pymunk_env.close()


def test_experimental_env_observation_values_are_valid():
    env = ExperimentalPymunkLunarLanderEnv()
    observation, _ = env.reset(seed=123)

    assert env.observation_space.contains(observation)
    assert np.isfinite(observation).all()


def test_experimental_env_seeded_resets_are_deterministic():
    first_env = ExperimentalPymunkLunarLanderEnv()
    second_env = ExperimentalPymunkLunarLanderEnv()

    first_observation, _ = first_env.reset(seed=123)
    second_observation, _ = second_env.reset(seed=123)

    assert np.array_equal(first_observation, second_observation)
    assert np.array_equal(
        first_env.demo.terrain.smooth_y,
        second_env.demo.terrain.smooth_y,
    )


def test_experimental_env_different_seeds_change_initial_state():
    first_env = ExperimentalPymunkLunarLanderEnv()
    second_env = ExperimentalPymunkLunarLanderEnv()

    first_observation, _ = first_env.reset(seed=123)
    second_observation, _ = second_env.reset(seed=456)

    assert not np.array_equal(first_observation, second_observation)
    assert not np.array_equal(
        first_env.demo.terrain.smooth_y,
        second_env.demo.terrain.smooth_y,
    )


def test_experimental_env_successive_unseeded_resets_advance_rng():
    env = ExperimentalPymunkLunarLanderEnv()

    first_observation, _ = env.reset(seed=123)
    second_observation, _ = env.reset()

    assert not np.array_equal(first_observation, second_observation)
    first_terrain = env.demo.terrain.smooth_y.copy()
    env.reset()
    assert not np.array_equal(first_terrain, env.demo.terrain.smooth_y)


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_experimental_env_accepts_all_discrete_actions(action):
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == (8,)
    assert np.isfinite(observation).all()
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert set(info) == {
        "termination_reason",
        "is_success",
        "inside_landing_zone",
    }


def test_action_spaces_match_box2d_contract():
    discrete_env = ExperimentalPymunkLunarLanderEnv()
    continuous_env = ExperimentalPymunkLunarLanderEnv(continuous=True)

    assert discrete_env.action_space == gym.spaces.Discrete(4)
    assert continuous_env.action_space == gym.spaces.Box(-1, 1, (2,), dtype=np.float32)


@pytest.mark.parametrize("gravity", [-12.0, 0.0, -12.1, 0.1, np.nan])
def test_invalid_gravity_matches_box2d_validation(gravity):
    with pytest.raises(AssertionError, match="must be between -12 and 0"):
        ExperimentalPymunkLunarLanderEnv(gravity=gravity)


def test_non_default_gravity_configures_pymunk_space():
    env = ExperimentalPymunkLunarLanderEnv(gravity=-4.0)
    env.reset(seed=123)

    assert tuple(env.demo.space.gravity) == (0.0, -4.0)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("wind_power", -0.1, "wind_power value is recommended"),
        ("wind_power", 20.1, "wind_power value is recommended"),
        ("turbulence_power", -0.1, "turbulence_power value is recommended"),
        ("turbulence_power", 2.1, "turbulence_power value is recommended"),
    ],
)
def test_wind_power_recommendations_match_box2d(argument, value, message):
    with pytest.warns(UserWarning, match=message):
        ExperimentalPymunkLunarLanderEnv(**{argument: value})


def test_continuous_zero_action_matches_discrete_noop():
    discrete_env = ExperimentalPymunkLunarLanderEnv()
    continuous_env = ExperimentalPymunkLunarLanderEnv(continuous=True)
    discrete_observation, _ = discrete_env.reset(seed=123)
    continuous_observation, _ = continuous_env.reset(seed=123)

    assert np.array_equal(discrete_observation, continuous_observation)
    discrete_step = discrete_env.step(0)
    continuous_step = continuous_env.step(np.zeros(2, dtype=np.float32))
    assert np.array_equal(discrete_step[0], continuous_step[0])
    assert discrete_step[1:] == continuous_step[1:]


def test_continuous_main_and_side_engines_apply_scaled_impulses():
    zero_demo = PymunkLunarLanderDemo(seed=123)
    engine_demo = PymunkLunarLanderDemo(seed=123)
    zero_demo.space.gravity = (0.0, 0.0)
    engine_demo.space.gravity = (0.0, 0.0)

    _, zero_main, zero_side = zero_demo._step_with_powers(
        np.array([0.0, 0.0]), continuous=True
    )
    _, main_power, side_power = engine_demo._step_with_powers(
        np.array([0.5, -0.75]), continuous=True
    )

    assert (zero_main, zero_side) == (0.0, 0.0)
    assert main_power == pytest.approx(0.75)
    assert side_power == pytest.approx(0.75)
    assert engine_demo.lander_body.velocity != zero_demo.lander_body.velocity
    assert engine_demo.lander_body.angular_velocity != pytest.approx(
        zero_demo.lander_body.angular_velocity
    )


def test_continuous_actions_are_clipped_like_box2d():
    clipped_env = ExperimentalPymunkLunarLanderEnv(continuous=True)
    bounded_env = ExperimentalPymunkLunarLanderEnv(continuous=True)
    clipped_env.reset(seed=123)
    bounded_env.reset(seed=123)

    clipped_step = clipped_env.step(np.array([2.0, -2.0], dtype=np.float32))
    bounded_step = bounded_env.step(np.array([1.0, -1.0], dtype=np.float32))

    assert np.array_equal(clipped_step[0], bounded_step[0])
    assert clipped_step[1:] == bounded_step[1:]


def test_continuous_engine_reward_uses_throttle_power():
    env = ExperimentalPymunkLunarLanderEnv(continuous=True)
    env.reset(seed=123)
    env.prev_shaping = float(
        -100 * np.linalg.norm(env._get_observation()[:2])
        - 100 * np.linalg.norm(env._get_observation()[2:4])
        - 100 * abs(env._get_observation()[4])
    )
    original_step = env.demo._step_with_powers
    env.demo._step_with_powers = lambda action, continuous: (
        env.demo.state(),
        0.75,
        0.75,
    )

    _, reward, _, _, _ = env.step(np.array([0.5, 0.75], dtype=np.float32))

    assert reward == pytest.approx(-(0.75 * 0.30 + 0.75 * 0.03))
    env.demo._step_with_powers = original_step


def test_wind_is_disabled_by_default():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)

    assert env.enable_wind is False
    assert not hasattr(env, "wind_idx")


def test_seeded_wind_is_deterministic_and_changes_trajectory():
    first = ExperimentalPymunkLunarLanderEnv(enable_wind=True)
    second = ExperimentalPymunkLunarLanderEnv(enable_wind=True)
    disabled = ExperimentalPymunkLunarLanderEnv(enable_wind=False)
    first.reset(seed=123)
    second.reset(seed=123)
    disabled.reset(seed=123)
    action = np.int64(0)

    for _ in range(20):
        first_observation = first.step(action)[0]
        second_observation = second.step(action)[0]
        disabled_observation = disabled.step(action)[0]

    assert np.array_equal(first_observation, second_observation)
    assert not np.array_equal(first_observation, disabled_observation)


def test_configurable_wind_and_turbulence_change_their_respective_motion():
    calm = ExperimentalPymunkLunarLanderEnv(
        enable_wind=True, wind_power=0.0, turbulence_power=0.0
    )
    windy = ExperimentalPymunkLunarLanderEnv(
        enable_wind=True, wind_power=7.0, turbulence_power=0.75
    )
    calm.reset(seed=123)
    windy.reset(seed=123)

    for _ in range(10):
        calm_observation = calm.step(0)[0]
        windy_observation = windy.step(0)[0]

    assert windy.wind_power == 7.0
    assert windy.turbulence_power == 0.75
    assert windy_observation[2] != pytest.approx(calm_observation[2])
    assert windy_observation[5] != pytest.approx(calm_observation[5])


def test_wind_uses_force_and_torque_units_before_physics_integration():
    env = ExperimentalPymunkLunarLanderEnv(
        enable_wind=True, wind_power=7.0, turbulence_power=0.75
    )
    env.reset(seed=123)
    env.wind_idx = 25
    env.torque_idx = -40
    env.demo.lander_body.force = (0.0, 0.0)
    env.demo.lander_body.torque = 0.0

    env._apply_wind()

    expected_force = (
        math.tanh(math.sin(0.02 * 25) + math.sin(math.pi * 0.01 * 25)) * 7.0
    )
    expected_torque = (
        math.tanh(math.sin(0.02 * -40) + math.sin(math.pi * 0.01 * -40)) * 0.75
    )
    assert tuple(env.demo.lander_body.force) == pytest.approx((expected_force, 0.0))
    assert env.demo.lander_body.torque == pytest.approx(expected_torque)


def test_experimental_env_crash_termination():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    env.demo.lander_body.position = (
        env.demo.world_width / 2.0,
        env.demo.terrain.helipad_y + 0.05,
    )
    env.demo.lander_body.velocity = (0.0, 0.0)
    env.demo.left_leg_body.position = (0.0, env.demo.world_height)
    env.demo.right_leg_body.position = (env.demo.world_width, env.demo.world_height)

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == -100.0
    assert info == {
        "termination_reason": "crash",
        "is_success": False,
        "inside_landing_zone": False,
    }


def test_experimental_env_stable_landing_termination():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    place_in_resting_pose(env.demo)

    for _ in range(100):
        observation, reward, terminated, truncated, info = env.step(0)
        if terminated:
            break

    assert terminated
    assert not truncated
    assert reward == 100.0
    assert info == {
        "termination_reason": "stable_landing",
        "is_success": True,
        "inside_landing_zone": True,
    }
    group_is_sleeping = all(
        body.is_sleeping
        for body in (
            env.demo.lander_body,
            env.demo.left_leg_body,
            env.demo.right_leg_body,
        )
    )
    assert group_is_sleeping or env.stable_landing_steps >= STABLE_LANDING_STEPS
    assert np.hypot(observation[2], observation[3]) < 0.01
    assert abs(observation[5]) < 0.01
    assert observation[6] == 1.0
    assert observation[7] == 1.0
    assert abs(observation[1]) < 0.02
    diagnostics = physics_diagnostics(env.demo, 0)
    for body_name in ("hull", "left_leg", "right_leg"):
        assert diagnostics[f"{body_name}_linear_speed"] <= STABLE_LINEAR_SPEED_THRESHOLD
        assert (
            diagnostics[f"{body_name}_angular_speed"] <= STABLE_ANGULAR_SPEED_THRESHOLD
        )

    ground_y = float(env.demo.terrain.smooth_y[CHUNKS // 2])
    for leg in (env.demo.left_leg_body, env.demo.right_leg_body):
        shape = next(iter(leg.shapes))
        lowest_point = min(
            leg.local_to_world(vertex).y for vertex in shape.get_vertices()
        )
        assert lowest_point >= ground_y - 0.02


def test_stability_fallback_terminates_when_native_sleep_is_disabled():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    place_in_resting_pose(env.demo)
    env.demo.space.sleep_time_threshold = float("inf")

    for _ in range(100):
        observation, reward, terminated, truncated, info = env.step(0)
        if terminated:
            break

    assert terminated
    assert not truncated
    assert reward == 100.0
    assert info["termination_reason"] == "stable_landing"
    assert not env.demo.lander_body.is_sleeping
    assert env.stable_landing_steps >= STABLE_LANDING_STEPS
    assert np.hypot(observation[2], observation[3]) < 0.01
    assert abs(observation[5]) < 0.01


def test_stability_counter_rejects_motion_and_resets_immediately():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    bodies = (
        env.demo.lander_body,
        env.demo.left_leg_body,
        env.demo.right_leg_body,
    )
    env.demo.leg_contacts[LEFT_LEG_COLLISION_TYPE] = 1
    env.demo.leg_contacts[RIGHT_LEG_COLLISION_TYPE] = 1
    for body in bodies:
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0

    for _ in range(STABLE_LANDING_STEPS - 1):
        assert not env._update_stable_landing_counter()
    assert env.stable_landing_steps == STABLE_LANDING_STEPS - 1

    env.demo.left_leg_body.velocity = (STABLE_LINEAR_SPEED_THRESHOLD * 2, 0.0)
    assert not env._update_stable_landing_counter()
    assert env.stable_landing_steps == 0

    env.demo.left_leg_body.velocity = (0.0, 0.0)
    env._update_stable_landing_counter()
    env.demo.leg_contacts[LEFT_LEG_COLLISION_TYPE] = 0
    assert not env._update_stable_landing_counter()
    assert env.stable_landing_steps == 0


def test_resting_hull_height_matches_scaled_box2d_geometry():
    demo = PymunkLunarLanderDemo(seed=123)
    place_in_resting_pose(demo)
    for _ in range(100):
        demo.step(0)
        if demo.lander_body.is_sleeping:
            break

    relative_angle = 0.4 - 0.05
    anchor_height = pymunk.Vec2d(-LEG_AWAY, LEG_DOWN).rotated(relative_angle).y
    foot_height = pymunk.Vec2d(0.0, -LEG_HEIGHT / 2).rotated(relative_angle).y
    expected_hull_y = (
        float(demo.terrain.smooth_y[CHUNKS // 2]) + anchor_height - foot_height
    )

    assert demo.lander_body.is_sleeping
    assert demo.lander_body.position.y == pytest.approx(expected_hull_y, abs=0.03)


def test_experimental_env_viewport_exit_termination_reason():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    env.demo.lander_body.position = (env.demo.world_width + 1.0, env.demo.world_height)
    env.demo.lander_body.velocity = (0.0, 0.0)

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == -100.0
    assert info == {
        "termination_reason": "viewport_exit",
        "is_success": False,
        "inside_landing_zone": False,
    }


@pytest.mark.parametrize("direction", [-1, 1])
@requires_box2d
def test_viewport_exit_boundary_and_timing_match_box2d(direction):
    box_env = gym.make("LunarLander-v3", disable_env_checker=True)
    pymunk_env = ExperimentalPymunkLunarLanderEnv()
    box_env.reset(seed=123)
    pymunk_env.reset(seed=123)
    target_x = 10.0 + direction * 10.01
    box_delta = target_x - box_env.unwrapped.lander.position.x
    for body in [box_env.unwrapped.lander, *box_env.unwrapped.legs]:
        body.position = (body.position.x + box_delta, body.position.y)
        body.linearVelocity = (0.0, 0.0)
    pymunk_delta = target_x - pymunk_env.demo.lander_body.position.x
    for body in (
        pymunk_env.demo.lander_body,
        pymunk_env.demo.left_leg_body,
        pymunk_env.demo.right_leg_body,
    ):
        body.position = (body.position.x + pymunk_delta, body.position.y)
        body.velocity = (0.0, 0.0)
    box_env.unwrapped.world.gravity = (0.0, 0.0)
    pymunk_env.demo.space.gravity = (0.0, 0.0)

    box_observation, box_reward, box_terminated, _, _ = box_env.step(0)
    pymunk_observation, pymunk_reward, pymunk_terminated, _, pymunk_info = (
        pymunk_env.step(0)
    )

    assert abs(box_observation[0]) >= 1.0
    assert abs(pymunk_observation[0]) >= 1.0
    assert box_terminated and pymunk_terminated
    assert box_reward == pymunk_reward == -100.0
    assert pymunk_info["termination_reason"] == "viewport_exit"
    box_env.close()
    pymunk_env.close()


@requires_box2d
def test_one_body_random_actions_isolate_center_of_mass_torque_mismatch():
    from scripts.analyze_lunar_lander_angular_dynamics import box_impulse

    world = Box2D.b2World(gravity=(0.0, -10.0))
    box_body = world.CreateDynamicBody(position=(10.0, 13.333333))
    box_body.CreatePolygonFixture(
        vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY],
        density=5.0,
    )
    space = pymunk.Space()
    space.gravity = (0.0, -10.0)
    pymunk_body = pymunk.Body(box_body.mass, box_body.inertia)
    pymunk_body.position = (10.0, 13.333333)
    shape = pymunk.Poly(pymunk_body, [(x / SCALE, y / SCALE) for x, y in LANDER_POLY])
    pymunk_body.center_of_gravity = shape.center_of_gravity
    pymunk_body.position = (10.0, 13.333333)
    space.add(pymunk_body, shape)
    baseline_space = pymunk.Space()
    baseline_space.gravity = (0.0, -10.0)
    baseline_body = pymunk.Body(box_body.mass, box_body.inertia)
    baseline_body.position = (10.0, 13.333333)
    baseline_shape = pymunk.Poly(
        baseline_body, [(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
    )
    baseline_space.add(baseline_body, baseline_shape)
    actions = np.random.default_rng(123).integers(0, 4, size=200)
    dispersions = np.random.default_rng(456).uniform(
        -1 / SCALE, 1 / SCALE, size=(200, 2)
    )

    for action, dispersion in zip(actions, dispersions, strict=True):
        offset, impulse, _ = box_impulse(
            int(action), float(box_body.angle), box_body.position, dispersion
        )
        if action:
            point = np.asarray(box_body.position) + offset
            box_body.ApplyLinearImpulse(tuple(impulse), tuple(point), True)
            pymunk_body.apply_impulse_at_world_point(
                tuple(impulse), tuple(np.asarray(pymunk_body.position) + offset)
            )
            baseline_body.apply_impulse_at_world_point(
                tuple(impulse), tuple(np.asarray(baseline_body.position) + offset)
            )
        world.Step(1 / 50, 180, 60)
        space.step(1 / 50)
        baseline_space.step(1 / 50)

    assert np.asarray(pymunk_body.velocity) == pytest.approx(
        np.asarray(box_body.linearVelocity), abs=1e-4
    )
    corrected_error = abs(pymunk_body.angular_velocity - box_body.angularVelocity)
    baseline_error = abs(baseline_body.angular_velocity - box_body.angularVelocity)
    assert corrected_error < baseline_error
    assert box_body.localCenter.y == pytest.approx(0.10130719)
    assert shape.center_of_gravity.y == pytest.approx(box_body.localCenter.y)
    assert pymunk_body.center_of_gravity.y == pytest.approx(box_body.localCenter.y)


def test_direct_env_has_no_internal_time_limit_truncation():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    env.demo._step_with_powers = lambda action, continuous: (
        env.demo.state(),
        0.0,
        0.0,
    )

    for _ in range(1001):
        _, _, terminated, truncated, info = env.step(0)

    assert not terminated
    assert not truncated
    assert info == {
        "termination_reason": None,
        "is_success": False,
        "inside_landing_zone": False,
    }


def test_experimental_env_render_returns_rgb_array():
    env = ExperimentalPymunkLunarLanderEnv(render_mode="rgb_array")
    env.reset(seed=123)

    frame = env.render()

    assert frame.shape == (400, 600, 3)
    assert frame.dtype == np.uint8
    assert frame[:50].mean() > frame[-50:].mean()
    assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] >= 5


def test_experimental_env_render_does_not_change_state():
    env = ExperimentalPymunkLunarLanderEnv(render_mode="rgb_array")
    observation, _ = env.reset(seed=123)
    body_positions = [tuple(body.position) for body in env.demo.space.bodies]
    rng_state = json.dumps(env.np_random.bit_generator.state, sort_keys=True)
    previous_shaping = env.prev_shaping
    stable_landing_steps = env.stable_landing_steps

    first_frame = env.render()
    second_frame = env.render()
    next_observation = env._get_observation()

    assert np.array_equal(observation, next_observation)
    assert np.array_equal(first_frame, second_frame)
    assert [tuple(body.position) for body in env.demo.space.bodies] == body_positions
    assert json.dumps(env.np_random.bit_generator.state, sort_keys=True) == rng_state
    assert env.prev_shaping == previous_shaping
    assert env.stable_landing_steps == stable_landing_steps


def test_render_calls_do_not_change_future_trajectory_or_rewards():
    rendered_env = ExperimentalPymunkLunarLanderEnv(render_mode="rgb_array")
    control_env = ExperimentalPymunkLunarLanderEnv(render_mode="rgb_array")
    rendered_env.reset(seed=123)
    control_env.reset(seed=123)

    for action in [0, 2, 1, 0, 3, 2, 0] * 3:
        rendered_env.render()
        rendered_env.render()
        rendered_step = rendered_env.step(action)
        control_step = control_env.step(action)

        assert np.array_equal(rendered_step[0], control_step[0])
        assert rendered_step[1:] == control_step[1:]


@pytest.mark.parametrize(
    ("action", "color", "horizontal_direction"),
    [
        (2, (255, 120, 20), 0),
        (1, (255, 150, 30), 1),
        (3, (255, 150, 30), -1),
    ],
)
def test_engine_actions_render_flames_in_plausible_regions(
    action, color, horizontal_direction
):
    env = ExperimentalPymunkLunarLanderEnv(render_mode="rgb_array")
    env.reset(seed=123)
    for body in env.demo.space.bodies:
        body.position = (body.position.x, body.position.y - 2.0)
    env.step(action)

    frame = env.render()
    flame_y, flame_x = np.nonzero(np.all(frame == color, axis=2))
    hull_screen = env._world_to_screen(tuple(body_origin_world(env.demo.lander_body)))

    assert flame_x.size >= 5
    assert abs(float(flame_x.mean()) - hull_screen[0]) < 100
    assert abs(float(flame_y.mean()) - hull_screen[1]) < 100
    if action == 2:
        assert flame_y.mean() > hull_screen[1]
    else:
        assert np.sign(flame_x.mean() - hull_screen[0]) == horizontal_direction


def test_render_without_configured_mode_warns_and_returns_none():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)

    with pytest.warns(UserWarning, match="without specifying any render mode"):
        assert env.render() is None


def test_human_mode_initializes_updates_and_renders_automatically(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.display.quit()
    env = ExperimentalPymunkLunarLanderEnv(render_mode="human")
    render_calls = 0
    original_render = env.render

    def tracked_render():
        nonlocal render_calls
        render_calls += 1
        return original_render()

    monkeypatch.setattr(env, "render", tracked_render)

    env.reset(seed=123)
    assert render_calls == 1
    assert pygame.display.get_init()
    assert env._screen is not None
    assert env._clock is not None
    assert env.render() is None
    env.step(0)
    assert render_calls == 3
    env.close()


def test_multiple_environments_and_reset_after_close_are_safe(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    first = ExperimentalPymunkLunarLanderEnv(render_mode="human")
    second = ExperimentalPymunkLunarLanderEnv(render_mode="human")
    first.reset(seed=123)
    second.reset(seed=456)

    first.close()
    first.close()
    assert second.render() is None
    second.close()
    observation, _ = first.reset(seed=789)

    assert observation in first.observation_space
    assert first.render() is None
    first.close()


def test_experimental_env_render_works_after_step():
    env = ExperimentalPymunkLunarLanderEnv(render_mode="rgb_array")
    env.reset(seed=123)
    env.step(2)

    frame = env.render()

    assert frame.shape == (400, 600, 3)
    assert frame.dtype == np.uint8


def test_experimental_env_close_is_idempotent():
    env = ExperimentalPymunkLunarLanderEnv(render_mode="rgb_array")
    env.reset(seed=123)
    env.render()

    env.close()
    env.close()


@requires_box2d
def test_hard_leg_impact_matches_box2d_termination_semantics():
    """Leg-only contact must not terminate unless the hull crashes."""
    box_env = gym.make("LunarLander-v3", disable_env_checker=True)
    pymunk_env = ExperimentalPymunkLunarLanderEnv()

    box_env.reset(seed=123)
    pymunk_env.reset(seed=123)

    box = box_env.unwrapped
    pymunk_demo = pymunk_env.demo

    box.world.gravity = (0.0, 0.0)
    pymunk_demo.space.gravity = (0.0, 0.0)

    box.game_over = False
    pymunk_demo.crashed = False

    for leg in box.legs:
        leg.ground_contact = True

    pymunk_demo.leg_contacts[LEFT_LEG_COLLISION_TYPE] = 1
    pymunk_demo.leg_contacts[RIGHT_LEG_COLLISION_TYPE] = 1

    box.lander.linearVelocity = (0.0, -4.0)
    pymunk_demo.lander_body.velocity = (0.0, -4.0)

    _, _, box_terminated, _, _ = box_env.step(0)
    _, _, pymunk_terminated, _, pymunk_info = pymunk_env.step(0)

    assert box_terminated == pymunk_terminated
    assert not pymunk_terminated
    assert pymunk_info["termination_reason"] is None

    box_env.close()
    pymunk_env.close()
