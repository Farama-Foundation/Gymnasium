"""A set of tests to help the designer of gymnasium environments verify that they work correctly."""

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
    skip_render: bool = False,
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
        skip_render: If `True` it does not check for equivalent renders. note:the render checked are automatically skipped if `render_mode` is not set or is "human".
        info_comparison: The options are
            If "equivalence" then checks if the `info`s are identical,
            If "superset" checks if `info_b` is a (non-strict) superset of `info_a`
            If "keys-equivalence" checks if the `info`s keys are identical (while ignoring the values).
            If "keys-superset" checks if the `info_b`s keys are a superset of `info_a`'s keys.
            If "skip" no checks are made at the `info`.
    """
    skip_render = (
        skip_render
        or env_a.unwrapped.render_mode in [None, "human"]
        or env_b.unwrapped.render in [None, "human"]
    )

    assert info_comparison in [
        "equivalence",
        "superset",
        "skip",
        "keys-equivalence",
        "keys-superset",
    ]

    assert env_a.action_space == env_b.action_space
    assert skip_obs or env_b.observation_space == env_b.observation_space

    env_a.action_space.seed(seed)
    obs_a, info_a = env_a.reset(seed=seed)
    obs_b, info_b = env_b.reset(seed=seed)

    assert skip_obs or data_equivalence(
        obs_a, obs_b
    ), f"resetting observation is not equivalent, observation_a = {obs_a}, observation_b = {obs_b}"
    if info_comparison == "equivalence":
        assert data_equivalence(
            info_a, info_b
        ), f"resetting info is not equivalent, info_a = {info_a}, info_b = {info_b}"
    elif info_comparison == "superset":
        for key in info_a:
            assert data_equivalence(
                info_a[key], info_b[key]
            ), f"resetting info is not a superset, key {key} present in info_a with value = {info_a[key]}, in info_b with value = {info_b[key]}"
    elif info_comparison == "keys-equivalance":
        assert (
            info_a.keys() == info_b.keys()
        ), f"resetting info keys are not equivalent, info_a's keys are {info_a.keys()}, info_b's keys are {info_b.keys()}"
    elif info_comparison == "keys-superset":
        assert (
            info_b.keys() >= info_a.keys()
        ), f"resetting info keys are not a superset, keys not present in info_b are: {info_b.keys() - info_a.keys()}"

    if not skip_render:
        assert (
            env_a.render() == env_b.render()
        ).all(), "resetting render is not equivalent"

    for step in range(num_steps):
        action = env_a.action_space.sample()
        obs_a, rew_a, terminal_a, truncated_a, info_a = env_a.step(action)
        obs_b, rew_b, terminal_b, truncated_b, info_b = env_b.step(action)
        assert skip_obs or data_equivalence(
            obs_a, obs_b
        ), f"stepping observation is not equivalent in step = {step}, observation_a = {obs_a}, observation_b = {obs_b}"
        assert skip_rew or data_equivalence(
            rew_a, rew_b
        ), f"stepping reward is not equivalent in step = {step}, reward_a = {rew_a}, reward_b = {rew_b}"
        assert (
            skip_terminal or terminal_a == terminal_b
        ), f"stepping terminal is not equivalent in step = {step}, terminal_a = {terminal_a}, terminal_b = {terminal_b}"
        assert (
            skip_truncated or truncated_a == truncated_b
        ), f"stepping truncated is not equivalent in step = {step}, truncated_a = {truncated_a}, truncated_b = {truncated_b}"
        if info_comparison == "equivalence":
            assert data_equivalence(
                info_a, info_b
            ), f"stepping info is not equivalent in step = {step}, info_a = {info_a}, info_b = {info_b}"
        elif info_comparison == "superset":
            for key in info_a:
                assert data_equivalence(
                    info_a[key], info_b[key]
                ), f"stepping info is not a superset in step = {step}, key {key} present in info_a with value = {info_a[key]}, in info_b with value = {info_b[key]}"
        elif info_comparison == "keys-equivalance":
            assert (
                info_a.keys() == info_b.keys()
            ), f"stepping info keys are not equivalent in step = {step}, info_a's keys are {info_a.keys()}, info_b's keys are {info_b.keys()}"
        elif info_comparison == "keys-superset":
            assert (
                info_b.keys() >= info_a.keys()
            ), f"stepping info keys are not a superset in step = {step}, keys not present in info_b are: {info_b.keys() - info_a.keys()}"
        if not skip_render:
            assert (
                env_a.render() == env_b.render()
            ).all(), "stepping render is not equivalent in step = {step}"

        if terminal_a or truncated_a or terminal_b or truncated_b:
            obs_a, info_a = env_a.reset(seed=seed)
            obs_b, info_b = env_b.reset(seed=seed)
            assert skip_obs or data_equivalence(
                obs_a, obs_b
            ), f"resetting observation is not equivalent in step = {step}, observation_a = {obs_a}, observation_b = {obs_b}"
            if info_comparison == "equivalence":
                assert data_equivalence(
                    info_a, info_b
                ), f"resetting info is not equivalent in step = {step}, info_a = {info_a}, info_b = {info_b}"
            elif info_comparison == "superset":
                for key in info_a:
                    assert data_equivalence(
                        info_a[key], info_b[key]
                    ), f"resetting info is not a superset in step = {step}, key {key} present in info_a with value = {info_a[key]}, in info_b with value = {info_b[key]}"
            elif info_comparison == "keys-equivalance":
                assert (
                    info_a.keys() == info_b.keys()
                ), f"resetting info keys are not equivalent in step = {step}, info_a's keys are {info_a.keys()}, info_b's keys are {info_b.keys()}"
            elif info_comparison == "keys-superset":
                assert (
                    info_b.keys() >= info_a.keys()
                ), f"resetting info keys are not a superset in step = {step}, keys not present in info_b are: {info_b.keys() - info_a.keys()}"
            if not skip_render:
                assert (
                    env_a.render() == env_b.render()
                ).all(), "resetting render is not equivalent in step = {step}"
