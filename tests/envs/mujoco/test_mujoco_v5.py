import warnings

import mujoco
import numpy as np
import pytest

import gymnasium as gym
from gymnasium.error import Error
from gymnasium.utils.env_match import check_environments_match


ALL_MUJOCO_ENVS = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "Humanoid",
    "HumanoidStandup",
    "InvertedDoublePendulum",
    "InvertedPendulum",
    "Pusher",
    "Reacher",
    "Swimmer",
    "Walker2d",
]


# Note: "HumnanoidStandup-v4" does not have `info`
# Note: "Humnanoid-v4/3" & "Ant-v4/3" fail this test
@pytest.mark.parametrize(
    "env_id",
    [
        "Ant-v5",
        "HalfCheetah-v5",
        "HalfCheetah-v4",
        "HalfCheetah-v3",
        "Hopper-v5",
        "Hopper-v4",
        "Hopper-v3",
        "Humanoid-v5",
        "HumanoidStandup-v5",
        "Swimmer-v5",
        "Swimmer-v4",
        "Swimmer-v3",
        "Walker2d-v5",
        "Walker2d-v4",
        "Walker2d-v3",
    ],
)
def test_verify_info_x_position(env_id: str):
    """Asserts that the environment has position[0] == info['x_position']."""
    env = gym.make(env_id, exclude_current_positions_from_observation=False)

    _, _ = env.reset()
    obs, _, _, _, info = env.step(env.action_space.sample())

    assert obs[0] == info["x_position"]


# Note: "HumnanoidStandup-v4" does not have `info`
# Note: "Humnanoid-v4/3" & "Ant-v4/3" fail this test
@pytest.mark.parametrize(
    "env_id",
    [
        "Ant-v5",
        "Humanoid-v5",
        "HumanoidStandup-v5",
        "Swimmer-v5",
        "Swimmer-v4",
        "Swimmer-v3",
    ],
)
def test_verify_info_y_position(env_id: str):
    """Asserts that the environment has position[1] == info['y_position']."""
    env = gym.make(env_id, exclude_current_positions_from_observation=False)

    _, _ = env.reset()
    obs, _, _, _, info = env.step(env.action_space.sample())

    assert obs[1] == info["y_position"]


# Note: "HumnanoidStandup-v4" does not have `info`
@pytest.mark.parametrize("env", ["HalfCheetah", "Hopper", "Swimmer", "Walker2d"])
@pytest.mark.parametrize("version", ["v5", "v4", "v3"])
def test_verify_info_x_velocity(env: str, version: str):
    """Asserts that the environment `info['x_velocity']` is properly assigned."""
    env = gym.make(f"{env}-{version}")
    env.reset()

    old_x = env.unwrapped.data.qpos[0]
    _, _, _, _, info = env.step(env.action_space.sample())
    new_x = env.unwrapped.data.qpos[0]

    dx = new_x - old_x
    vel_x = dx / env.dt
    assert vel_x == info["x_velocity"]


# Note: "HumnanoidStandup-v4" does not have `info`
@pytest.mark.parametrize("env_id", ["Swimmer-v5", "Swimmer-v4", "Swimmer-v3"])
def test_verify_info_y_velocity(env_id: str):
    """Asserts that the environment `info['y_velocity']` is properly assigned."""
    env = gym.make(env_id)
    env.reset()

    old_y = env.unwrapped.data.qpos[1]
    _, _, _, _, info = env.step(env.action_space.sample())
    new_y = env.unwrapped.data.qpos[1]

    dy = new_y - old_y
    vel_y = dy / env.dt
    assert vel_y == info["y_velocity"]


@pytest.mark.parametrize("env_id", ["Ant-v5", "Ant-v4", "Ant-v3"])
def test_verify_info_xy_velocity_xpos(env_id: str):
    """Asserts that the environment `info['x/y_velocity']` is properly assigned, for the ant environment which uses kinmatics for the velocity."""
    env = gym.make(env_id)
    env.reset()

    old_xy = env.get_body_com("torso")[:2].copy()
    _, _, _, _, info = env.step(env.action_space.sample())
    new_xy = env.get_body_com("torso")[:2].copy()

    dxy = new_xy - old_xy
    vel_x, vel_y = dxy / env.dt
    assert vel_x == info["x_velocity"]
    assert vel_y == info["y_velocity"]


@pytest.mark.parametrize("env_id", ["Humanoid-v5", "Humanoid-v4", "Humanoid-v3"])
def test_verify_info_xy_velocity_com(env_id: str):
    """Asserts that the environment `info['x/y_velocity']` is properly assigned, for the humanoid environment which uses kinmatics of Center Of Mass for the velocity."""

    def mass_center(model, data):
        mass = np.expand_dims(model.body_mass, axis=1)
        xpos = data.xipos
        return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

    env = gym.make(env_id)
    env.reset()

    old_xy = mass_center(env.unwrapped.model, env.unwrapped.data)
    _, _, _, _, info = env.step(env.action_space.sample())
    new_xy = mass_center(env.unwrapped.model, env.unwrapped.data)

    dxy = new_xy - old_xy
    vel_x, vel_y = dxy / env.dt
    assert vel_x == info["x_velocity"]
    assert vel_y == info["y_velocity"]


# Note: Hopper-v4/3/2 does not have `info['reward_survive']`, but it is still affected
# Note: Walker2d-v4/3/2 does not have `info['reward_survive']`, but it is still affected
# Note: Inverted(Double)Pendulum-v4/2 does not have `info['reward_survive']`, but it is still affected
# Note: all `v4/v3/v2` environments with a heathly reward are fail this test
@pytest.mark.parametrize(
    "env",
    [
        "Ant",
        "Hopper",
        "Humanoid",
        "InvertedDoublePendulum",
        "InvertedPendulum",
        "Walker2d",
    ],
)
@pytest.mark.parametrize("version", ["v5"])
def test_verify_reward_survive(env: str, version: str):
    """Assert that `reward_survive` is 0 on `terminal` states and not 0 on non-`terminal` states."""
    env = gym.make(f"{env}-{version}", reset_noise_scale=0)
    env.reset(seed=0)
    env.action_space.seed(0)

    for step in range(175):
        obs, rew, terminal, truncated, info = env.step(env.action_space.sample())

        if terminal:
            assert info["reward_survive"] == 0
            break

        assert info["reward_survive"] != 0

    assert (
        terminal
    ), "The environment, should have terminated, if not the test is not valid."


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]


@pytest.mark.parametrize("env", ALL_MUJOCO_ENVS)
@pytest.mark.parametrize("version", ["v5"])
@pytest.mark.parametrize("frame_skip", [1, 2, 3, 4, 5])
def test_frame_skip(env: str, version: str, frame_skip: int):
    """Verify that all `mujoco` envs work with different `frame_skip` values."""
    env_id = f"{env}-{version}"
    env = gym.make(env_id, frame_skip=frame_skip)

    # Test if env adheres to Gym API
    with warnings.catch_warnings(record=True) as w:
        gym.utils.env_checker.check_env(env.unwrapped, skip_render_check=True)
        env.close()
    for warning in w:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning.message}")


# Dev Note: This can be version env parametrized because each env has it's own reward function
@pytest.mark.parametrize("version", ["v5"])
def test_reward_sum(version: str):
    """Assert that the total reward equals the sum of the individual reward terms, also asserts that the reward function has no fp ordering arithmetic errors."""
    NUM_STEPS = 100
    env = gym.make(f"Ant-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert reward == (info["reward_survive"] + info["reward_forward"]) - (
            -info["reward_ctrl"] + -info["reward_contact"]
        )

    env = gym.make(f"HalfCheetah-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert reward == info["reward_forward"] + info["reward_ctrl"]

    env = gym.make(f"Hopper-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert (
            reward
            == info["reward_forward"] + info["reward_survive"] + info["reward_ctrl"]
        )

    env = gym.make(f"Humanoid-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert reward == (info["reward_forward"] + info["reward_survive"]) + (
            info["reward_ctrl"] + info["reward_contact"]
        )

    env = gym.make(f"HumanoidStandup-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert (
            reward
            == info["reward_linup"]
            + info["reward_quadctrl"]
            + info["reward_impact"]
            + 1
        )

    env = gym.make(f"InvertedDoublePendulum-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert (
            reward
            == info["reward_survive"]
            + info["distance_penalty"]
            + info["velocity_penalty"]
        )

    env = gym.make(f"InvertedPendulum-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert reward == info["reward_survive"]

    env = gym.make(f"Pusher-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert reward == info["reward_dist"] + info["reward_ctrl"] + info["reward_near"]

    env = gym.make(f"Reacher-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert reward == info["reward_dist"] + info["reward_ctrl"]

    env = gym.make(f"Swimmer-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert reward == info["reward_forward"] + info["reward_ctrl"]

    env = gym.make(f"Walker2d-{version}")
    env.reset()
    for _ in range(NUM_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert (
            reward
            == info["reward_forward"] + info["reward_survive"] + info["reward_ctrl"]
        )


# Note: the environtments that are not present, is because they do not have identical behaviour
def test_identical_behaviour_v45():
    """Verify that v4 -> v5 transition. Does not change the behaviour of the environments in any unexpected way."""
    NUM_STEPS = 100

    env_v4 = gym.make("Ant-v4")
    env_v5 = gym.make("Ant-v5", include_cfrc_ext_in_observation=False)
    check_environments_match(
        env_v4, env_v5, NUM_STEPS, skip_rew=True, info_comparison="skip"
    )

    env_v4 = gym.make("HalfCheetah-v4")
    env_v5 = gym.make("HalfCheetah-v5")
    check_environments_match(env_v4, env_v5, NUM_STEPS, info_comparison="skip")

    # env_v4 = gym.make("Hopper-v4")
    # env_v5 = gym.make("Hopper-v5")
    # check_environments_match(env_v4, env_v5, num_steps, skip_rew=true, info_comparison="superset")

    # skipping humanoid, everything has changed

    env_v4 = gym.make("HumanoidStandup-v4")
    env_v5 = gym.make("HumanoidStandup-v5")
    check_environments_match(
        env_v4, env_v5, NUM_STEPS, skip_obs=True, info_comparison="superset"
    )

    env_v4 = gym.make("InvertedDoublePendulum-v4")
    env_v5 = gym.make("InvertedDoublePendulum-v5")
    check_environments_match(
        env_v4,
        env_v5,
        NUM_STEPS,
        skip_obs=True,
        skip_rew=True,
        info_comparison="superset",
    )

    env_v4 = gym.make("InvertedPendulum-v4")
    env_v5 = gym.make("InvertedPendulum-v5")
    check_environments_match(
        env_v4, env_v5, NUM_STEPS, skip_rew=True, info_comparison="superset"
    )

    env_v4 = gym.make("Pusher-v4")
    env_v5 = gym.make("Pusher-v5")
    check_environments_match(env_v4, env_v5, NUM_STEPS, info_comparison="skip")

    env_v4 = gym.make("Reacher-v4")
    env_v5 = gym.make("Reacher-v5")
    check_environments_match(
        env_v4, env_v5, NUM_STEPS, skip_obs=True, info_comparison="skip"
    )

    env_v4 = gym.make("Swimmer-v4")
    env_v5 = gym.make("Swimmer-v5")
    check_environments_match(env_v4, env_v5, NUM_STEPS, info_comparison="skip")

    # skipping Walker2d, since the model has changed


@pytest.mark.parametrize("version", ["v5", "v4"])
def test_ant_com(version: str):
    """Verify the kinmatic behaviour of the ant."""
    env = gym.make(
        f"Ant-{version}"
    )  # `env` contains `data : MjData` and `model : MjModel`
    env.reset()  # randomly initlizies the `data.qpos` and `data.qvel`, calls mujoco.mj_forward(env.model, env.data)

    x_position_before = env.unwrapped.data.qpos[0]
    x_position_before_com = env.unwrapped.data.body("torso").xpos[0]
    assert x_position_before == x_position_before_com, "before failed"  # This succeeds

    random_control = env.action_space.sample()
    _, _, _, _, info = env.step(
        random_control
    )  # This calls mujoco.mj_step(env.model, env.data, nstep=env.frame_skip)
    mujoco.mj_kinematics(env.unwrapped.model, env.unwrapped.data)

    x_position_after = env.unwrapped.data.qpos[0]
    x_position_after_com = env.unwrapped.data.body("torso").xpos[0]
    assert x_position_after == x_position_after_com, "after failed"  # This succeeds


@pytest.mark.parametrize("version", ["v5", "v4", "v3", "v2"])
def test_set_state(version: str):
    """Simple Test to verify that `mujocoEnv.set_state()` works correctly."""
    env = gym.make(f"Hopper-{version}")
    env.reset()
    new_qpos = np.array(
        [0.00136962, 1.24769787, -0.00459026, -0.00483472, 0.0031327, 0.00412756]
    )
    new_qvel = np.array(
        [0.00106636, 0.00229497, 0.00043625, 0.00435072, 0.00315854, -0.00497261]
    )
    env.set_state(new_qpos, new_qvel)
    assert (env.unwrapped.data.qpos == new_qpos).all()
    assert (env.unwrapped.data.qvel == new_qvel).all()


# Note: HumanoidStandup-v4/v3 does not have `info`
# Note: Ant-v4/v3 fails this test
# Note: Humanoid-v4/v3 fails this test
# Note: v2 does not have `info`
@pytest.mark.parametrize(
    "env_id", ["Ant-v5", "Humanoid-v5", "Swimmer-v5", "Swimmer-v4", "Swimmer-v3"]
)
def test_distance_from_origin_info(env_id: str):
    """Verify that `info"distance_from_origin"` is correct."""
    env = gym.make(env_id)
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["distance_from_origin"] == np.linalg.norm(
        env.unwrapped.data.qpos[0:2] - env.init_qpos[0:2]
    )


@pytest.mark.parametrize("env", ["Hopper", "HumanoidStandup", "Walker2d"])
@pytest.mark.parametrize("version", ["v5"])
def test_z_distance_from_origin_info(env: str, version: str):
    """Verify that `info"z_distance_from_origin"` is correct."""
    env = gym.make(f"{env}-{version}")
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    mujoco.mj_kinematics(env.unwrapped.model, env.unwrapped.data)
    z_index = env.observation_structure["skipped_qpos"]
    assert (
        info["z_distance_from_origin"]
        == env.unwrapped.data.qpos[z_index] - env.init_qpos[z_index]
    )


@pytest.mark.parametrize(
    "env",
    [
        "Ant",
        "HalfCheetah",
        "Hopper",
        "Humanoid",
        "HumanoidStandup",
        "InvertedPendulum",
        "Swimmer",
        "Walker2d",
    ],
)
@pytest.mark.parametrize("version", ["v5"])
def test_observation_structure(env: str, version: str):
    """Verify that the `env.observation_structure` is properly defined."""
    env = gym.make(f"{env}-{version}")
    if hasattr(env, "observation_structure"):
        return

    obs_struct = env.observation_structure

    assert (
        env.unwrapped.model.nq == obs_struct.get("skipped_qpos", 0) + obs_struct["qpos"]
    )
    assert env.unwrapped.model.nv == obs_struct["qvel"]
    if obs_struct.get("cinert", False):
        assert (env.unwrapped.model.nbody - 1) * 10 == obs_struct["cinert"]
    if obs_struct.get("cvel", False):
        assert (env.unwrapped.model.nbody - 1) * 6 == obs_struct["cvel"]
    if obs_struct.get("qfrc_actuator", False):
        assert env.unwrapped.model.nv - 6 == obs_struct["qfrc_actuator"]
    if obs_struct.get("cfrc_ext", False):
        assert (env.unwrapped.model.nbody - 1) * 6 == obs_struct["cfrc_ext"]
    if obs_struct.get("ten_lenght", False):
        assert env.unwrapped.model.ntendon == obs_struct["ten_lenght"]
    if obs_struct.get("ten_velocity", False):
        assert env.unwrapped.model.ntendon == obs_struct["ten_velocity"]


@pytest.mark.parametrize(
    "env",
    [
        "Ant",
        "HalfCheetah",
        "Hopper",
        "Humanoid",
        "HumanoidStandup",
        # "InvertedDoublePendulum",
        # "InvertedPendulum",
        # "Pusher",
        # "Reacher",
        "Swimmer",
        "Walker2d",
    ],
)
@pytest.mark.parametrize("version", ["v5"])
def test_reset_info(env: str, version: str):
    """Verify that the environment returns info with `reset()`."""
    env = gym.make(f"{env}-{version}")
    _, reset_info = env.reset()
    assert len(reset_info) > 0


"""
[Bug Report] [Documentation] Inverted Double Pendulum max Height is wrong

The Documentation States:
```md
The maximum standing height of the system is 1.196 m when all the parts are perpendicularly vertical on top of each other)
```
but the height of each pole is 0.6 (0.6+0.6==1.2)
https://github.com/Farama-Foundation/Gymnasium/blob/deb50802facfd827abd4d1f0cf1069afb12a726b/gymnasium/envs/mujoco/assets/inverted_double_pendulum.xml#L33-L39
"""


# Note: the max height used to be wrong in the documentation.
@pytest.mark.parametrize("version", ["v5"])
def test_inverted_double_pendulum_max_height(version: str):
    """Verify the max height of Inverted Double Pendulum."""
    env = gym.make(f"InvertedDoublePendulum-{version}", reset_noise_scale=0)
    env.reset()
    y = env.unwrapped.data.site_xpos[0][2]
    assert y == 1.2


@pytest.mark.parametrize("version", ["v4"])
def test_inverted_double_pendulum_max_height_old(version: str):
    """Verify the max height of Inverted Double Pendulum (v4 does not have `reset_noise_scale` argument)."""
    env = gym.make(f"InvertedDoublePendulum-{version}")
    env.set_state(env.init_qpos, env.init_qvel)
    y = env.unwrapped.data.site_xpos[0][2]
    assert y == 1.2


# note: fails with `brax==0.9.0`
@pytest.mark.parametrize("version", ["v5", "v4"])
def test_model_object_count(version: str):
    """Verify that all the objects of the model are loaded, mostly useful for using non-mujoco simulator."""
    env = gym.make(f"Ant-{version}")
    assert env.unwrapped.model.nq == 15
    assert env.unwrapped.model.nv == 14
    assert env.unwrapped.model.nu == 8
    assert env.unwrapped.model.nbody == 14
    assert env.unwrapped.model.nbvh == 14
    assert env.unwrapped.model.njnt == 9
    assert env.unwrapped.model.ngeom == 14
    assert env.unwrapped.model.ntendon == 0

    env = gym.make(f"HalfCheetah-{version}")
    assert env.unwrapped.model.nq == 9
    assert env.unwrapped.model.nv == 9
    assert env.unwrapped.model.nu == 6
    assert env.unwrapped.model.nbody == 8
    assert env.unwrapped.model.nbvh == 10
    assert env.unwrapped.model.njnt == 9
    assert env.unwrapped.model.ngeom == 9
    assert env.unwrapped.model.ntendon == 0

    env = gym.make(f"Hopper-{version}")
    assert env.unwrapped.model.nq == 6
    assert env.unwrapped.model.nv == 6
    assert env.unwrapped.model.nu == 3
    assert env.unwrapped.model.nbody == 5
    assert env.unwrapped.model.nbvh == 5
    assert env.unwrapped.model.njnt == 6
    assert env.unwrapped.model.ngeom == 5
    assert env.unwrapped.model.ntendon == 0

    env = gym.make(f"Humanoid-{version}")
    assert env.unwrapped.model.nq == 24
    assert env.unwrapped.model.nv == 23
    assert env.unwrapped.model.nu == 17
    assert env.unwrapped.model.nbody == 14
    assert env.unwrapped.model.nbvh == 22
    assert env.unwrapped.model.njnt == 18
    assert env.unwrapped.model.ngeom == 18
    assert env.unwrapped.model.ntendon == 2

    env = gym.make(f"HumanoidStandup-{version}")
    assert env.unwrapped.model.nq == 24
    assert env.unwrapped.model.nv == 23
    assert env.unwrapped.model.nu == 17
    assert env.unwrapped.model.nbody == 14
    assert env.unwrapped.model.nbvh == 22
    assert env.unwrapped.model.njnt == 18
    assert env.unwrapped.model.ngeom == 18
    assert env.unwrapped.model.ntendon == 2

    env = gym.make(f"InvertedDoublePendulum-{version}")
    assert env.unwrapped.model.nq == 3
    assert env.unwrapped.model.nv == 3
    assert env.unwrapped.model.nu == 1
    assert env.unwrapped.model.nbody == 4
    assert env.unwrapped.model.nbvh == 6
    assert env.unwrapped.model.njnt == 3
    assert env.unwrapped.model.ngeom == 5
    assert env.unwrapped.model.ntendon == 0

    env = gym.make(f"InvertedPendulum-{version}")
    assert env.unwrapped.model.nq == 2
    assert env.unwrapped.model.nv == 2
    assert env.unwrapped.model.nu == 1
    assert env.unwrapped.model.nbody == 3
    assert env.unwrapped.model.nbvh == 3
    assert env.unwrapped.model.njnt == 2
    assert env.unwrapped.model.ngeom == 3
    assert env.unwrapped.model.ntendon == 0

    env = gym.make(f"Pusher-{version}")
    assert env.unwrapped.model.nq == 11
    assert env.unwrapped.model.nv == 11
    assert env.unwrapped.model.nu == 7
    assert env.unwrapped.model.nbody == 13
    assert env.unwrapped.model.nbvh == 18
    assert env.unwrapped.model.njnt == 11
    assert env.unwrapped.model.ngeom == 21
    assert env.unwrapped.model.ntendon == 0

    env = gym.make(f"Reacher-{version}")
    assert env.unwrapped.model.nq == 4
    assert env.unwrapped.model.nv == 4
    assert env.unwrapped.model.nu == 2
    assert env.unwrapped.model.nbody == 5
    assert env.unwrapped.model.nbvh == 5
    assert env.unwrapped.model.njnt == 4
    assert env.unwrapped.model.ngeom == 10
    assert env.unwrapped.model.ntendon == 0

    env = gym.make(f"Swimmer-{version}")
    assert env.unwrapped.model.nq == 5
    assert env.unwrapped.model.nv == 5
    assert env.unwrapped.model.nu == 2
    assert env.unwrapped.model.nbody == 4
    assert env.unwrapped.model.nbvh == 4
    assert env.unwrapped.model.njnt == 5
    assert env.unwrapped.model.ngeom == 4
    assert env.unwrapped.model.ntendon == 0

    env = gym.make(f"Walker2d-{version}")
    assert env.unwrapped.model.nq == 9
    assert env.unwrapped.model.nv == 9
    assert env.unwrapped.model.nu == 6
    assert env.unwrapped.model.nbody == 8
    assert env.unwrapped.model.nbvh == 8
    assert env.unwrapped.model.njnt == 9
    assert env.unwrapped.model.ngeom == 8
    assert env.unwrapped.model.ntendon == 0


"""
def test_dt():
    env_a = gym.make("Ant-v5", include_cfrc_ext_in_observation=False)
    env_b = gym.make("Ant-v5", include_cfrc_ext_in_observation=False, frame_skip=1)
    env_b.unwrapped.model.opt.timestep = 0.05

    assert env_a.dt == env_b.dt
    check_environments_match(env_a, env_b, num_steps=100)
"""
