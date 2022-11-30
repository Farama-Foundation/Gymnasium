import gymnasium as gym
from gymnasium.envs.registration import EnvSpec

# To ignore the trailing whitespaces, will need flake to ignore this file.
# flake8: noqa

reduced_registry = {
    env_id: env_spec
    for env_id, env_spec in gym.registry.items()
    if env_spec.entry_point != "shimmy.atari_env:AtariEnv"
}


def test_pprint_custom_registry():
    """Testing a registry different from default."""
    a = {
        "CartPole-v0": gym.envs.registry["CartPole-v0"],
        "CartPole-v1": gym.envs.registry["CartPole-v1"],
    }
    out = gym.pprint_registry(a, disable_print=True)

    correct_out = """===== classic_control =====
CartPole-v0 CartPole-v1 

"""
    assert out == correct_out
