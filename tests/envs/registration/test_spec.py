"""Tests that `gym.spec` works as expected."""

import re

import pytest

import gymnasium as gym


def test_spec():
    spec = gym.spec("CartPole-v1")
    assert spec.id == "CartPole-v1"
    assert spec is gym.envs.registry["CartPole-v1"]


def test_spec_missing_lookup():
    gym.register(id="TestEnv-v0", entry_point="no-entry-point")
    gym.register(id="TestEnv-v15", entry_point="no-entry-point")
    gym.register(id="TestEnv-v9", entry_point="no-entry-point")
    gym.register(id="OtherEnv-v100", entry_point="no-entry-point")

    with pytest.raises(
        gym.error.DeprecatedEnv,
        match=re.escape(
            "Environment version v1 for `TestEnv` is deprecated. Please use `TestEnv-v15` instead."
        ),
    ):
        gym.spec("TestEnv-v1")

    with pytest.raises(
        gym.error.UnregisteredEnv,
        match=re.escape(
            "Environment version `v1000` for environment `TestEnv` doesn't exist. It provides versioned environments: [ `v0`, `v9`, `v15` ]."
        ),
    ):
        gym.spec("TestEnv-v1000")

    with pytest.raises(
        gym.error.UnregisteredEnv,
        match=re.escape("Environment `UnknownEnv` doesn't exist."),
    ):
        gym.spec("UnknownEnv-v1")

    del gym.registry["TestEnv-v0"]
    del gym.registry["TestEnv-v15"]
    del gym.registry["TestEnv-v9"]
    del gym.registry["OtherEnv-v100"]


def test_spec_malformed_lookup():
    with pytest.raises(
        gym.error.Error,
        match=re.escape(
            "Malformed environment ID: “Breakout-v0”. (Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))"
        ),
    ):
        gym.spec("“Breakout-v0”")


def test_spec_versioned_lookups():
    gym.register("test/TestEnv-v5", "no-entry-point")

    with pytest.raises(
        gym.error.VersionNotFound,
        match=re.escape(
            "Environment version `v9` for environment `test/TestEnv` doesn't exist. It provides versioned environments: [ `v5` ]."
        ),
    ):
        gym.spec("test/TestEnv-v9")

    with pytest.raises(
        gym.error.DeprecatedEnv,
        match=re.escape(
            "Environment version v4 for `test/TestEnv` is deprecated. Please use `test/TestEnv-v5` instead."
        ),
    ):
        gym.spec("test/TestEnv-v4")

    assert gym.spec("test/TestEnv-v5") is not None
    del gym.registry["test/TestEnv-v5"]


def test_spec_default_lookups():
    gym.register("test/TestEnv", "no-entry-point")

    with pytest.raises(
        gym.error.DeprecatedEnv,
        match=re.escape(
            "Environment version `v0` for environment `test/TestEnv` doesn't exist. It provides the default version `test/TestEnv`."
        ),
    ):
        gym.spec("test/TestEnv-v0")

    assert gym.spec("test/TestEnv") is not None
    del gym.registry["test/TestEnv"]
