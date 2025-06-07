import collections
import warnings

import mujoco
import numpy as np
import pytest

import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.envs.mujoco.utils import check_mujoco_reset_state
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env
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
        "Hopper-v5",
        "Hopper-v4",
        "Humanoid-v5",
        "HumanoidStandup-v5",
        "Swimmer-v5",
        "Swimmer-v4",
        "Walker2d-v5",
        "Walker2d-v4",
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
    ],
)
def test_verify_info_y_position(env_id: str):
    """Asserts that the environment has position[1] == info['y_position']."""
    env = gym.make(env_id, exclude_current_positions_from_observation=False)

    _, _ = env.reset()
    obs, _, _, _, info = env.step(env.action_space.sample())

    assert obs[1] == info["y_position"]


# Note: "HumnanoidStandup-v4" does not have `info`
@pytest.mark.parametrize("env_name", ["HalfCheetah", "Hopper", "Swimmer", "Walker2d"])
@pytest.mark.parametrize("version", ["v5", "v4"])
def test_verify_info_x_velocity(env_name: str, version: str):
    """Asserts that the environment `info['x_velocity']` is properly assigned."""
    env = gym.make(f"{env_name}-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    env.reset()

    old_x = env.data.qpos[0]
    _, _, _, _, info = env.step(env.action_space.sample())
    new_x = env.data.qpos[0]

    dx = new_x - old_x
    vel_x = dx / env.dt
    assert vel_x == info["x_velocity"]


# Note: "HumnanoidStandup-v4" does not have `info`
@pytest.mark.parametrize("env_id", ["Swimmer-v5", "Swimmer-v4"])
def test_verify_info_y_velocity(env_id: str):
    """Asserts that the environment `info['y_velocity']` is properly assigned."""
    env = gym.make(env_id).unwrapped
    assert isinstance(env, (MujocoEnv))
    env.reset()

    old_y = env.data.qpos[1]
    _, _, _, _, info = env.step(env.action_space.sample())
    new_y = env.data.qpos[1]

    dy = new_y - old_y
    vel_y = dy / env.dt
    assert vel_y == info["y_velocity"]


@pytest.mark.parametrize("env_id", ["Ant-v5", "Ant-v4"])
def test_verify_info_xy_velocity_xpos(env_id: str):
    """Asserts that the environment `info['x/y_velocity']` is properly assigned, for the ant environment which uses kinmatics for the velocity."""
    env = gym.make(env_id).unwrapped
    assert isinstance(env, (MujocoEnv))
    env.reset()

    old_xy = env.get_body_com("torso")[:2].copy()
    _, _, _, _, info = env.step(env.action_space.sample())
    new_xy = env.get_body_com("torso")[:2].copy()

    dxy = new_xy - old_xy
    vel_x, vel_y = dxy / env.dt
    assert vel_x == info["x_velocity"]
    assert vel_y == info["y_velocity"]


@pytest.mark.parametrize("env_id", ["Humanoid-v5", "Humanoid-v4"])
def test_verify_info_xy_velocity_com(env_id: str):
    """Asserts that the environment `info['x/y_velocity']` is properly assigned, for the humanoid environment which uses kinmatics of Center Of Mass for the velocity."""

    def mass_center(model, data):
        mass = np.expand_dims(model.body_mass, axis=1)
        xpos = data.xipos
        return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

    env = gym.make(env_id).unwrapped
    assert isinstance(env, (MujocoEnv))
    env.reset()

    old_xy = mass_center(env.model, env.data)
    _, _, _, _, info = env.step(env.action_space.sample())
    new_xy = mass_center(env.model, env.data)

    dxy = new_xy - old_xy
    vel_x, vel_y = dxy / env.dt
    assert vel_x == info["x_velocity"]
    assert vel_y == info["y_velocity"]


# Note: Hopper-v4/3/2 does not have `info['reward_survive']`, but it is still affected
# Note: Walker2d-v4/3/2 does not have `info['reward_survive']`, but it is still affected
# Note: Inverted(Double)Pendulum-v4/2 does not have `info['reward_survive']`, but it is still affected
# Note: all `v4/v3/v2` environments with a heathly reward are fail this test
@pytest.mark.parametrize(
    "env_name",
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
def test_verify_reward_survive(env_name: str, version: str):
    """Assert that `reward_survive` is 0 on `terminal` states and not 0 on non-`terminal` states."""
    env = gym.make(f"{env_name}-{version}", reset_noise_scale=0).unwrapped
    assert isinstance(env, (MujocoEnv))
    env.reset(seed=0)
    env.action_space.seed(1)

    terminal = False
    for step in range(80):
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
        "A Box observation space maximum value is infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]


@pytest.mark.parametrize("env_name", ALL_MUJOCO_ENVS)
@pytest.mark.parametrize("version", ["v5"])
@pytest.mark.parametrize("frame_skip", [1, 2, 3, 4, 5])
def test_frame_skip(env_name: str, version: str, frame_skip: int):
    """Verify that all `mujoco` envs work with different `frame_skip` values."""
    env_id = f"{env_name}-{version}"
    env = gym.make(env_id, frame_skip=frame_skip)

    # Test if env adheres to Gym API
    with warnings.catch_warnings(record=True) as w:
        check_env(env.unwrapped, skip_render_check=True)
        env.close()
    for warning in w:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning.message}")


# Dev Note: This can not be version env parametrized because each env has it's own reward function
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


env_conf = collections.namedtuple("env_conf", "env_name, obs, rew, term, info")


# Note: the environments "HalfCheetah", "Pusher", "Swimmer", are identical between `v4` & `v5` (excluding `info`)
@pytest.mark.parametrize(
    "env_conf",
    [
        env_conf("Ant", True, True, False, "skip"),
        env_conf("HalfCheetah", False, False, False, "skip"),
        env_conf("Hopper", False, True, False, "superset"),
        # skipping humanoid, everything has changed
        env_conf("HumanoidStandup", True, False, False, "superset"),
        env_conf("InvertedDoublePendulum", True, True, False, "superset"),
        env_conf("InvertedPendulum", False, True, False, "superset"),
        env_conf("Pusher", True, True, False, "keys-superset"),  # pusher-v4
        env_conf("Reacher", True, True, False, "keys-equivalence"),
        env_conf("Swimmer", False, False, False, "skip"),
        env_conf("Walker2d", True, True, True, "keys-superset"),
    ],
)
def test_identical_behaviour_v45(env_conf, NUM_STEPS: int = 100):
    """Verify that v4 -> v5 transition. Does not change the behaviour of the environments in any unexpected way."""
    if env_conf.env_name == "Pusher" and mujoco.__version__ >= "3.0.0":
        pytest.skip("Pusher-v4 is not compatible with mujoco >= 3")

    env_v4 = gym.make(f"{env_conf.env_name}-v4")
    env_v5 = gym.make(f"{env_conf.env_name}-v5")

    check_environments_match(
        env_v4,
        env_v5,
        NUM_STEPS,
        skip_obs=env_conf.obs,
        skip_rew=env_conf.rew,
        skip_terminal=env_conf.term,
        info_comparison=env_conf.info,
    )


@pytest.mark.parametrize("version", ["v5", "v4"])
def test_ant_com(version: str):
    """Verify the kinmatic behaviour of the ant."""
    # `env` contains `data : MjData` and `model : MjModel`
    env = gym.make(f"Ant-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    env.reset()  # randomly initlizies the `data.qpos` and `data.qvel`, calls mujoco.mj_forward(env.model, env.data)

    x_position_before = env.data.qpos[0]
    x_position_before_com = env.data.body("torso").xpos[0]
    assert x_position_before == x_position_before_com, "before failed"  # This succeeds

    random_control = env.action_space.sample()
    # This calls mujoco.mj_step(env.model, env.data, nstep=env.frame_skip)
    _, _, _, _, info = env.step(random_control)
    mujoco.mj_kinematics(env.model, env.data)

    x_position_after = env.data.qpos[0]
    x_position_after_com = env.data.body("torso").xpos[0]
    assert x_position_after == x_position_after_com, "after failed"  # This succeeds


@pytest.mark.parametrize("version", ["v5", "v4"])
def test_set_state(version: str):
    """Simple Test to verify that `mujocoEnv.set_state()` works correctly."""
    env = gym.make(f"Hopper-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    env.reset()
    new_qpos = np.array(
        [0.00136962, 1.24769787, -0.00459026, -0.00483472, 0.0031327, 0.00412756]
    )
    new_qvel = np.array(
        [0.00106636, 0.00229497, 0.00043625, 0.00435072, 0.00315854, -0.00497261]
    )
    env.set_state(new_qpos, new_qvel)
    assert (env.data.qpos == new_qpos).all()
    assert (env.data.qvel == new_qvel).all()


# Note: HumanoidStandup-v4/v3 does not have `info`
# Note: Ant-v4/v3 fails this test
# Note: Humanoid-v4/v3 fails this test
# Note: v2 does not have `info`
@pytest.mark.parametrize(
    "env_id", ["Ant-v5", "Humanoid-v5", "Swimmer-v5", "Swimmer-v4"]
)
def test_distance_from_origin_info(env_id: str):
    """Verify that `info"distance_from_origin"` is correct."""
    env = gym.make(env_id).unwrapped
    assert isinstance(env, (MujocoEnv))
    env.reset()

    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["distance_from_origin"] == np.linalg.norm(
        env.data.qpos[0:2] - env.init_qpos[0:2]
    )


@pytest.mark.parametrize("env_name", ["Hopper", "HumanoidStandup", "Walker2d"])
@pytest.mark.parametrize("version", ["v5"])
def test_z_distance_from_origin_info(env_name: str, version: str):
    """Verify that `info["z_distance_from_origin"]` is correct."""
    env = gym.make(f"{env_name}-{version}").unwrapped
    assert isinstance(env, MujocoEnv)
    env.reset()

    _, _, _, _, info = env.step(env.action_space.sample())
    mujoco.mj_kinematics(env.model, env.data)
    z_index = env.observation_structure["skipped_qpos"]
    assert (
        info["z_distance_from_origin"]
        == env.data.qpos[z_index] - env.init_qpos[z_index]
    )


@pytest.mark.parametrize("env_name", ALL_MUJOCO_ENVS)
@pytest.mark.parametrize("version", ["v5"])
def test_observation_structure(env_name: str, version: str):
    """Verify that the `env.observation_structure` is properly defined."""
    env = gym.make(f"{env_name}-{version}").unwrapped
    assert isinstance(env, MujocoEnv)
    if not hasattr(env, "observation_structure"):
        pytest.skip("Environment doesn't have an `observation_structure` attribute")

    obs_struct = env.observation_structure

    assert env.model.nq == obs_struct.get("skipped_qpos", 0) + obs_struct["qpos"]
    assert env.model.nv == obs_struct["qvel"]
    if obs_struct.get("cinert", False):
        assert (env.model.nbody - 1) * 10 == obs_struct["cinert"]
    if obs_struct.get("cvel", False):
        assert (env.model.nbody - 1) * 6 == obs_struct["cvel"]
    if obs_struct.get("qfrc_actuator", False):
        assert env.model.nv - 6 == obs_struct["qfrc_actuator"]
    if obs_struct.get("cfrc_ext", False):
        assert (env.model.nbody - 1) * 6 == obs_struct["cfrc_ext"]
    if obs_struct.get("ten_lenght", False):
        assert env.model.ntendon == obs_struct["ten_lenght"]
    if obs_struct.get("ten_velocity", False):
        assert env.model.ntendon == obs_struct["ten_velocity"]


@pytest.mark.parametrize(
    "env_name",
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
def test_reset_info(env_name: str, version: str):
    """Verify that the environment returns info with `reset()`."""
    env = gym.make(f"{env_name}-{version}")
    _, reset_info = env.reset()
    assert len(reset_info) > 0


# Note: the max height used to be wrong in the documentation. (1.196m instead of 1.2m)
@pytest.mark.parametrize("version", ["v5"])
def test_inverted_double_pendulum_max_height(version: str):
    """Verify the max height of Inverted Double Pendulum."""
    env = gym.make(f"InvertedDoublePendulum-{version}", reset_noise_scale=0).unwrapped
    assert isinstance(env, (MujocoEnv))
    env.reset()

    y = env.data.site_xpos[0][2]
    assert y == 1.2


@pytest.mark.parametrize("version", ["v4"])
def test_inverted_double_pendulum_max_height_old(version: str):
    """Verify the max height of Inverted Double Pendulum (v4 does not have `reset_noise_scale` argument)."""
    env = gym.make(f"InvertedDoublePendulum-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    env.set_state(env.init_qpos, env.init_qvel)

    y = env.data.site_xpos[0][2]
    assert y == 1.2


# note: fails with `brax==0.9.0`
@pytest.mark.parametrize("version", ["v5", "v4"])
def test_model_object_count(version: str):
    """Verify that all the objects of the model are loaded, mostly useful for using non-mujoco simulator."""
    env = gym.make(f"Ant-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    assert env.model.nq == 15
    assert env.model.nv == 14
    assert env.model.nu == 8
    assert env.model.nbody == 14
    assert env.model.nbvh == 14
    assert env.model.njnt == 9
    assert env.model.ngeom == 14
    assert env.model.ntendon == 0

    env = gym.make(f"HalfCheetah-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    assert env.model.nq == 9
    assert env.model.nv == 9
    assert env.model.nu == 6
    assert env.model.nbody == 8
    assert env.model.nbvh == 10
    assert env.model.njnt == 9
    assert env.model.ngeom == 9
    assert env.model.ntendon == 0

    env = gym.make(f"Hopper-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    assert env.model.nq == 6
    assert env.model.nv == 6
    assert env.model.nu == 3
    assert env.model.nbody == 5
    assert env.model.nbvh == 5
    assert env.model.njnt == 6
    assert env.model.ngeom == 5
    assert env.model.ntendon == 0

    env = gym.make(f"Humanoid-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    assert env.model.nq == 24
    assert env.model.nv == 23
    assert env.model.nu == 17
    assert env.model.nbody == 14
    assert env.model.nbvh == 22
    assert env.model.njnt == 18
    assert env.model.ngeom == 18
    assert env.model.ntendon == 2

    env = gym.make(f"HumanoidStandup-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    assert env.model.nq == 24
    assert env.model.nv == 23
    assert env.model.nu == 17
    assert env.model.nbody == 14
    assert env.model.nbvh == 22
    assert env.model.njnt == 18
    assert env.model.ngeom == 18
    assert env.model.ntendon == 2

    env = gym.make(f"InvertedDoublePendulum-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    assert env.model.nq == 3
    assert env.model.nv == 3
    assert env.model.nu == 1
    assert env.model.nbody == 4
    assert env.model.nbvh == 6
    assert env.model.njnt == 3
    assert env.model.ngeom == 5
    assert env.model.ntendon == 0

    env = gym.make(f"InvertedPendulum-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    assert env.model.nq == 2
    assert env.model.nv == 2
    assert env.model.nu == 1
    assert env.model.nbody == 3
    assert env.model.nbvh == 3
    assert env.model.njnt == 2
    assert env.model.ngeom == 3
    assert env.model.ntendon == 0

    if not (version == "v4" and mujoco.__version__ >= "3.0.0"):
        env = gym.make(f"Pusher-{version}").unwrapped
        assert isinstance(env, (MujocoEnv))
        assert env.model.nq == 11
        assert env.model.nv == 11
        assert env.model.nu == 7
        assert env.model.nbody == 13
        if mujoco.__version__ >= "3.1.4":
            assert env.model.nbvh == 7
        elif mujoco.__version__ >= "3.1.2":
            assert env.model.nbvh == 8
        else:
            assert env.model.nbvh == 18
        assert env.model.njnt == 11
        if version == "v4":
            assert env.model.ngeom == 21
        else:
            assert env.model.ngeom == 20
        assert env.model.ntendon == 0

    env = gym.make(f"Reacher-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    assert env.model.nq == 4
    assert env.model.nv == 4
    assert env.model.nu == 2
    assert env.model.nbody == 5
    if mujoco.__version__ >= "3.1.2":
        assert env.model.nbvh == 3
    assert env.model.njnt == 4
    assert env.model.ngeom == 10
    assert env.model.ntendon == 0

    env = gym.make(f"Swimmer-{version}").unwrapped
    assert isinstance(env, (MujocoEnv))
    assert env.model.nq == 5
    assert env.model.nv == 5
    assert env.model.nu == 2
    assert env.model.nbody == 4
    if mujoco.__version__ >= "3.1.2":
        assert env.model.nbvh == 0
    assert env.model.njnt == 5
    assert env.model.ngeom == 4
    assert env.model.ntendon == 0

    env = gym.make(f"Walker2d-{version}").unwrapped
    assert isinstance(env, MujocoEnv)
    assert env.model.nq == 9
    assert env.model.nv == 9
    assert env.model.nu == 6
    assert env.model.nbody == 8
    assert env.model.nbvh == 8
    assert env.model.njnt == 9
    assert env.model.ngeom == 8
    assert env.model.ntendon == 0


# note: fails with `mujoco-mjx==3.0.1`
@pytest.mark.parametrize("version", ["v5", "v4"])
def test_model_sensors(version: str):
    """Verify that all the sensors of the model are loaded."""
    env = gym.make(f"Ant-{version}").unwrapped
    assert env.data.cfrc_ext.shape == (14, 6)

    env = gym.make(f"Humanoid-{version}").unwrapped
    assert env.data.cinert.shape == (14, 10)
    assert env.data.cvel.shape == (14, 6)
    assert env.data.qfrc_actuator.shape == (23,)
    assert env.data.cfrc_ext.shape == (14, 6)

    if version != "v3":  # HumanoidStandup v3 does not exist
        env = gym.make(f"HumanoidStandup-{version}").unwrapped
        assert env.data.cinert.shape == (14, 10)
        assert env.data.cvel.shape == (14, 6)
        assert env.data.qfrc_actuator.shape == (23,)
        assert env.data.cfrc_ext.shape == (14, 6)


def test_dt():
    """Assert that env.dt gets assigned correctly."""
    env_a = gym.make("Ant-v5", include_cfrc_ext_in_observation=False).unwrapped
    env_b = gym.make(
        "Ant-v5", include_cfrc_ext_in_observation=False, frame_skip=1
    ).unwrapped
    assert isinstance(env_a, MujocoEnv)
    assert isinstance(env_b, MujocoEnv)
    env_b.model.opt.timestep = 0.05

    assert env_a.dt == env_b.dt
    # check_environments_match(env_a, env_b, num_steps=100)   # This Fails as expected


@pytest.mark.parametrize(
    "env_id",
    [
        "Ant-v5",
        "Ant-v4",
        "HalfCheetah-v5",
        "HalfCheetah-v4",
        "Hopper-v5",
        "Hopper-v4",
        "Humanoid-v5",
        "Humanoid-v4",
        "HumanoidStandup-v5",
        "InvertedDoublePendulum-v5",
        "InvertedPendulum-v5",
        "Swimmer-v5",
        "Swimmer-v4",
        "Walker2d-v5",
        "Walker2d-v4",
    ],
)
def test_reset_noise_scale(env_id):
    """Checks that when `reset_noise_scale=0` we have deterministic initialization."""
    env = gym.make(env_id, reset_noise_scale=0).unwrapped
    env.reset()

    assert np.all(env.data.qpos == env.init_qpos)
    assert np.all(env.data.qvel == env.init_qvel)


@pytest.mark.parametrize("env_name", ALL_MUJOCO_ENVS)
@pytest.mark.parametrize("version", ["v5", "v4"])
def test_reset_state(env_name: str, version: str):
    """Asserts that `reset()` properly resets the internal state."""
    if env_name == "Pusher" and version == "v4" and mujoco.__version__ >= "3.0.0":
        pytest.skip("Skipping Pusher-v4 as not compatible with mujoco >= 3.0")

    env = gym.make(f"{env_name}-{version}")
    check_mujoco_reset_state(env)
