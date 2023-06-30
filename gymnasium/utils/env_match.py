"""A set of tests to help the desiner of gymansium environments verify that they work correctly."""

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence


def check_environments_match(
    env_a: gym.Env,
    env_b: gym.Env,
    num_steps: int,
    seed: int = 0,
    skip_obs: bool = False,
    skip_rew: bool = False,
    skip_terminal: bool = False,
    skip_truncated: bool = False,
    info_comparison: str = "equivalence",
):
    """Checks if the environments `env_a` & `env_b` are identical.

    Args:
        env_a: First environment to check.
        env_b: Second environment to check.
        num_steps: number of timesteps to test for, setting to 0 tests only resetting.
        seed: used the seed the reset & actions.
        skip_obs: If `True` it does not check for equivalence of the observation.
        skip_rew: If `True` it does not check for equivalence of the observation.
        skip_terminal: If `True` it does not check for equivalence of the observation.
        skip_truncated: If `True` it does not check for equivalence of the observation.
        skip_info: If `True` it does not check for equivalence of the observation.
        info_comparison: If "equivalence" then checks if the `info`s are identical,
            if "superset" checks if `info_b` is a (non-strict) superset of `info_a`
            if "skip" no checks are made at the `info`.
    """
    assert info_comparison in ["equivalence", "superset", "skip"]

    assert env_a.action_space == env_b.action_space
    assert skip_obs or env_b.observation_space == env_b.observation_space

    env_a.action_space.seed(seed)
    obs_a, info_a = env_a.reset(seed=seed)
    obs_b, info_b = env_b.reset(seed=seed)

    assert skip_obs or data_equivalence(
        obs_a, obs_b
    ), "resetting observation is not equivalent"
    if info_comparison == "equivalence":
        assert data_equivalence(info_a, info_b), "resetting info is not equivalent"
    elif info_comparison == "superset":
        for key in info_a:
            assert data_equivalence(
                info_a[key], info_b[key]
            ), "resetting info is not a superset"

    for _ in range(num_steps):
        action = env_a.action_space.sample()
        obs_a, rew_a, terminal_a, truncated_a, info_a = env_a.step(action)
        obs_b, rew_b, terminal_b, truncated_b, info_b = env_b.step(action)
        assert skip_obs or data_equivalence(
            obs_a, obs_b
        ), "stepping observation is not equivalent"
        assert skip_rew or data_equivalence(
            rew_a, rew_b
        ), "stepping reward is not equivalent"
        assert (
            skip_terminal or terminal_a == terminal_b
        ), "stepping terminal is not equivalent"
        assert (
            skip_truncated or truncated_a == truncated_b
        ), "stepping truncated is not equivalent"
        if info_comparison == "equivalence":
            assert data_equivalence(info_a, info_b), "stepping info is not equivalent"
        elif info_comparison == "superset":
            for key in info_a:
                assert data_equivalence(
                    info_a[key], info_b[key]
                ), "stepping info is not a superset"

        if terminal_a or truncated_a or terminal_b or truncated_b:
            obs_a, info_a = env_a.reset(seed=seed)
            obs_b, info_b = env_b.reset(seed=seed)
            assert skip_obs or data_equivalence(
                obs_a, obs_b
            ), "resetting observation is not equivalent"
            if info_comparison == "equivalence":
                assert data_equivalence(
                    info_a, info_b
                ), "resetting info is not equivalent"
            elif info_comparison == "superset":
                for key in info_a:
                    assert data_equivalence(
                        info_a[key], info_b[key]
                    ), "resetting info is not a superset"
