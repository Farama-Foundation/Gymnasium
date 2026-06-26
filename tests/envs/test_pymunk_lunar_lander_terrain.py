import numpy as np
import pytest

pytest.importorskip("pymunk")

from scripts.pymunk_lunar_lander_terrain import (  # noqa: E402
    CHUNKS,
    PymunkLunarLanderDemo,
)


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


def test_main_thrust_reduces_downward_velocity():
    passive_demo = PymunkLunarLanderDemo(seed=123)
    thrust_demo = PymunkLunarLanderDemo(seed=123)

    passive_state = passive_demo.step(0)
    thrust_state = thrust_demo.step(2)

    assert thrust_state.velocity_y > passive_state.velocity_y


def test_orientation_actions_produce_opposite_angular_acceleration():
    left_demo = PymunkLunarLanderDemo(seed=123)
    right_demo = PymunkLunarLanderDemo(seed=123)

    left_state = left_demo.step(1)
    right_state = right_demo.step(3)

    assert left_state.angular_velocity < 0
    assert right_state.angular_velocity > 0


def test_leg_contacts_become_active_after_normal_landing():
    demo = PymunkLunarLanderDemo(seed=42)

    for _ in range(200):
        state = demo.step(0)
        if state.left_leg_contact and state.right_leg_contact:
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
