import pytest

from gymnasium.envs.registration import EnvSpec
from tests.envs.utils import mujoco_testing_env_specs


EPS = 1e-6

EXCLUDE_POS_FROM_OBS = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "Humanoid",
    "Swimmer",
    "Walker2d",
]


@pytest.mark.parametrize(
    "env_spec",
    mujoco_testing_env_specs,
    ids=[env_spec.id for env_spec in mujoco_testing_env_specs],
)
def test_obs_space_mujoco_environments(env_spec: EnvSpec):
    """Check that the returned observations are contained in the observation space of the environment"""
    env = env_spec.make(disable_env_checker=True)
    reset_obs, info = env.reset()
    assert env.observation_space.contains(
        reset_obs
    ), f"Observation returned by reset() of {env_spec.id} is not contained in the default observation space {env.observation_space}."

    action = env.action_space.sample()
    step_obs, _, _, _, _ = env.step(action)
    assert env.observation_space.contains(
        step_obs
    ), f"Observation returned by step(action) of {env_spec.id} is not contained in the default observation space {env.observation_space}."

    if env_spec.name in EXCLUDE_POS_FROM_OBS and (
        env_spec.version == 4 or env_spec.version == 3
    ):
        env = env_spec.make(
            disable_env_checker=True, exclude_current_positions_from_observation=False
        )
        reset_obs, info = env.reset()
        assert env.observation_space.contains(
            reset_obs
        ), f"Observation of {env_spec.id} is not contained in the default observation space {env.observation_space} when excluding current position from observation."

        step_obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(
            step_obs
        ), f"Observation returned by step(action) of {env_spec.id} is not contained in the default observation space {env.observation_space} when excluding current position from observation."

    # Ant-v4 has the option of including contact forces in the observation space with the use_contact_forces argument
    if env_spec.name == "Ant" and env_spec.version == 4:
        env = env_spec.make(disable_env_checker=True, use_contact_forces=True)
        reset_obs, info = env.reset()
        assert env.observation_space.contains(
            reset_obs
        ), f"Observation of {env_spec.id} is not contained in the default observation space {env.observation_space} when using contact forces."

        step_obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(
            step_obs
        ), f"Observation returned by step(action) of {env_spec.id} is not contained in the default observation space {env.observation_space} when using contact forces."
