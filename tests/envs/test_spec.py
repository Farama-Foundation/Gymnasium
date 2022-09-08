"""Tests that gymnasium.spec works as expected."""

import re

import pytest

import gymnasium


def test_spec():
    spec = gymnasium.spec("CartPole-v1")
    assert spec.id == "CartPole-v1"
    assert spec is gymnasium.envs.registry["CartPole-v1"]


def test_spec_kwargs():
    map_name_value = "8x8"
    env = gymnasium.make("FrozenLake-v1", map_name=map_name_value)
    assert env.spec.kwargs["map_name"] == map_name_value


def test_spec_missing_lookup():
    gymnasium.register(id="Test1-v0", entry_point="no-entry-point")
    gymnasium.register(id="Test1-v15", entry_point="no-entry-point")
    gymnasium.register(id="Test1-v9", entry_point="no-entry-point")
    gymnasium.register(id="Other1-v100", entry_point="no-entry-point")

    with pytest.raises(
        gymnasium.error.DeprecatedEnv,
        match=re.escape(
            "Environment version v1 for `Test1` is deprecated. Please use `Test1-v15` instead."
        ),
    ):
        gymnasium.spec("Test1-v1")

    with pytest.raises(
        gymnasium.error.UnregisteredEnv,
        match=re.escape(
            "Environment version `v1000` for environment `Test1` doesn't exist. It provides versioned environments: [ `v0`, `v9`, `v15` ]."
        ),
    ):
        gymnasium.spec("Test1-v1000")

    with pytest.raises(
        gymnasium.error.UnregisteredEnv,
        match=re.escape("Environment Unknown1 doesn't exist. "),
    ):
        gymnasium.spec("Unknown1-v1")


def test_spec_malformed_lookup():
    with pytest.raises(
        gymnasium.error.Error,
        match=f'^{re.escape("Malformed environment ID: “Breakout-v0”.(Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))")}$',
    ):
        gymnasium.spec("“Breakout-v0”")


def test_spec_versioned_lookups():
    gymnasium.register("test/Test2-v5", "no-entry-point")

    with pytest.raises(
        gymnasium.error.VersionNotFound,
        match=re.escape(
            "Environment version `v9` for environment `test/Test2` doesn't exist. It provides versioned environments: [ `v5` ]."
        ),
    ):
        gymnasium.spec("test/Test2-v9")

    with pytest.raises(
        gymnasium.error.DeprecatedEnv,
        match=re.escape(
            "Environment version v4 for `test/Test2` is deprecated. Please use `test/Test2-v5` instead."
        ),
    ):
        gymnasium.spec("test/Test2-v4")

    assert gymnasium.spec("test/Test2-v5") is not None


def test_spec_default_lookups():
    gymnasium.register("test/Test3", "no-entry-point")

    with pytest.raises(
        gymnasium.error.DeprecatedEnv,
        match=re.escape(
            "Environment version `v0` for environment `test/Test3` doesn't exist. It provides the default version test/Test3`."
        ),
    ):
        gymnasium.spec("test/Test3-v0")

    assert gymnasium.spec("test/Test3") is not None
