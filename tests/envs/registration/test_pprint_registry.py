from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.registration import EnvSpec


# To ignore the trailing whitespaces, will need flake to ignore this file.
# flake8: noqa

EXAMPLE_ENTRY_POINT = "gymnasium.envs.classic_control.cartpole:CartPoleEnv"


def test_pprint_default_registry():
    out = gym.pprint_registry(disable_print=True)
    assert isinstance(out, str) and len(out) > 0


def test_pprint_example_registry():
    """Testing a registry different from default."""
    example_registry: dict[str, EnvSpec] = {
        "CartPole-v0": EnvSpec("CartPole-v0", EXAMPLE_ENTRY_POINT),
        "CartPole-v1": EnvSpec("CartPole-v1", EXAMPLE_ENTRY_POINT),
        "CartPole-v2": EnvSpec("CartPole-v2", EXAMPLE_ENTRY_POINT),
        "CartPole-v3": EnvSpec("CartPole-v3", EXAMPLE_ENTRY_POINT),
    }

    out = gym.pprint_registry(example_registry, disable_print=True)
    correct_out = """===== classic_control =====
CartPole-v0 CartPole-v1 CartPole-v2
CartPole-v3"""
    assert out == correct_out


def test_pprint_namespace():
    example_registry: dict[str, EnvSpec] = {
        "CartPole-v0": EnvSpec(
            "CartPole-v0", "gymnasium.envs.classic_control.cartpole:CartPoleEnv"
        ),
        "CartPole-v1": EnvSpec(
            "CartPole-v1", "gymnasium.envs.classic_control:CartPoleEnv"
        ),
        "CartPole-v2": EnvSpec("CartPole-v2", "gymnasium.cartpole:CartPoleEnv"),
        "CartPole-v3": EnvSpec("CartPole-v3", lambda: CartPoleEnv()),
        "ExampleNamespace/CartPole-v2": EnvSpec(
            "ExampleNamespace/CartPole-v2", "gymnasium.envs.classic_control:CartPoleEnv"
        ),
    }

    out = gym.pprint_registry(example_registry, disable_print=True)
    correct_out = """===== classic_control =====
CartPole-v0 CartPole-v1
===== cartpole =====
CartPole-v2
===== None =====
CartPole-v3
===== ExampleNamespace =====
ExampleNamespace/CartPole-v2"""
    assert out == correct_out


def test_pprint_n_columns():
    example_registry = {
        "CartPole-v0": EnvSpec("CartPole-v0", EXAMPLE_ENTRY_POINT),
        "CartPole-v1": EnvSpec("CartPole-v1", EXAMPLE_ENTRY_POINT),
        "CartPole-v2": EnvSpec("CartPole-v2", EXAMPLE_ENTRY_POINT),
        "CartPole-v3": EnvSpec("CartPole-v3", EXAMPLE_ENTRY_POINT),
    }

    out = gym.pprint_registry(example_registry, num_cols=2, disable_print=True)
    correct_out = """===== classic_control =====
CartPole-v0 CartPole-v1
CartPole-v2 CartPole-v3"""
    assert out == correct_out

    out = gym.pprint_registry(example_registry, num_cols=5, disable_print=True)
    correct_out = """===== classic_control =====
CartPole-v0 CartPole-v1 CartPole-v2 CartPole-v3"""
    assert out == correct_out


def test_pprint_exclude_namespace():
    example_registry: dict[str, EnvSpec] = {
        "Test/CartPole-v0": EnvSpec("Test/CartPole-v0", EXAMPLE_ENTRY_POINT),
        "Test/CartPole-v1": EnvSpec("Test/CartPole-v1", EXAMPLE_ENTRY_POINT),
        "CartPole-v2": EnvSpec("CartPole-v2", EXAMPLE_ENTRY_POINT),
        "CartPole-v3": EnvSpec("CartPole-v3", EXAMPLE_ENTRY_POINT),
    }

    out = gym.pprint_registry(
        example_registry, exclude_namespaces=["Test"], disable_print=True
    )
    correct_out = """===== classic_control =====
CartPole-v2 CartPole-v3"""
    assert out == correct_out

    out = gym.pprint_registry(
        example_registry, exclude_namespaces=["classic_control"], disable_print=True
    )
    correct_out = """===== Test =====
Test/CartPole-v0 Test/CartPole-v1"""
    assert out == correct_out

    example_registry["Example/CartPole-v4"] = EnvSpec(
        "Example/CartPole-v4", EXAMPLE_ENTRY_POINT
    )
    out = gym.pprint_registry(
        example_registry, exclude_namespaces=["Test", "Example"], disable_print=True
    )
    correct_out = """===== classic_control =====
CartPole-v2 CartPole-v3"""
    assert out == correct_out
