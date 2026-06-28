import numpy as np
import pytest

pytest.importorskip("pymunk")

from scripts.pymunk_lunar_lander_terrain import (  # noqa: E402
    CHUNKS,
    LEFT_LEG_COLLISION_TYPE,
    MAX_EPISODE_STEPS,
    RIGHT_LEG_COLLISION_TYPE,
    STABLE_LANDING_STEPS,
    ExperimentalPymunkLunarLanderEnv,
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
    assert info == {"termination_reason": None, "is_success": False}


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


def test_experimental_env_successive_unseeded_resets_advance_rng():
    env = ExperimentalPymunkLunarLanderEnv()

    first_observation, _ = env.reset(seed=123)
    second_observation, _ = env.reset()

    assert not np.array_equal(first_observation, second_observation)


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
    assert set(info) == {"termination_reason", "is_success"}


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
    assert info == {"termination_reason": "crash", "is_success": False}


def test_experimental_env_stable_landing_termination():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    env.demo.space.gravity = (0.0, 0.0)
    env.demo.lander_body.position = (
        env.demo.world_width / 2.0,
        env.demo.terrain.helipad_y + 1.0,
    )
    env.demo.lander_body.velocity = (0.0, 0.0)
    env.demo.lander_body.angle = 0.0
    env.demo.lander_body.angular_velocity = 0.0
    env.demo.leg_contacts[LEFT_LEG_COLLISION_TYPE] = 1
    env.demo.leg_contacts[RIGHT_LEG_COLLISION_TYPE] = 1

    for _ in range(STABLE_LANDING_STEPS + 30):
        _, reward, terminated, truncated, info = env.step(0)
        if terminated:
            break

    assert terminated
    assert not truncated
    assert reward == 100.0
    assert info == {"termination_reason": "stable_landing", "is_success": True}


def test_experimental_env_viewport_exit_termination_reason():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    env.demo.lander_body.position = (env.demo.world_width + 1.0, env.demo.world_height)
    env.demo.lander_body.velocity = (0.0, 0.0)

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == -100.0
    assert info == {"termination_reason": "viewport_exit", "is_success": False}


def test_experimental_env_time_limit_truncation():
    env = ExperimentalPymunkLunarLanderEnv()
    env.reset(seed=123)
    env.demo.space.gravity = (0.0, 0.0)
    env.demo.lander_body.position = (env.demo.world_width / 2.0, env.demo.world_height)
    env.demo.lander_body.velocity = (0.0, 0.0)

    for _ in range(MAX_EPISODE_STEPS):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break

    assert not terminated
    assert truncated
    assert info == {"termination_reason": "time_limit", "is_success": False}
