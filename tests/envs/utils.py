"""Finds all the specs that we can test with"""

import gymnasium as gym
from gymnasium import logger
from gymnasium.envs.registration import EnvSpec


def try_make_env(env_spec: EnvSpec) -> gym.Env | None:
    """Tries to make the environment showing if it is possible.

    Warning the environments have no wrappers, including time limit and order enforcing.
    """
    # To avoid issues with registered environments during testing, we check that the spec entry points are from gymnasium.envs.
    if (
        isinstance(env_spec.entry_point, str)
        and "gymnasium.envs." in env_spec.entry_point
    ):
        try:
            return env_spec.make(disable_env_checker=True).unwrapped
        except (
            ImportError,
            AttributeError,
            gym.error.DependencyNotInstalled,
            gym.error.MissingArgument,
        ) as e:
            logger.warn(f"Not testing {env_spec.id} due to error: {e}")
    return None


# Tries to make all environment to test with
all_testing_initialised_envs: list[gym.Env | None] = [
    try_make_env(env_spec) for env_spec in gym.envs.registry.values()
]
all_testing_initialised_envs: list[gym.Env] = [
    env for env in all_testing_initialised_envs if env is not None
]

# All testing, mujoco and gymnasium environment specs
all_testing_env_specs: list[EnvSpec] = [
    env.spec for env in all_testing_initialised_envs
]
mujoco_testing_env_specs: list[EnvSpec] = [
    env_spec
    for env_spec in all_testing_env_specs
    if "gymnasium.envs.mujoco" in env_spec.entry_point
]
gym_testing_env_specs: list[EnvSpec] = [
    env_spec
    for env_spec in all_testing_env_specs
    if any(
        f"gymnasium.envs.{ep}" in env_spec.entry_point
        for ep in ["box2d", "classic_control", "toy_text"]
    )
]
