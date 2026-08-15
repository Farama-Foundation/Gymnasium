"""Test suite for the constructor arguments that wrappers save for `Wrapper.spec`."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

import gymnasium as gym
from gymnasium.utils import RecordConstructorArgs

# `vector` is the sub-module of vector wrappers, not a wrapper. Vector wrappers are
# excluded because `VectorWrapper.spec` forwards the spec of the wrapped environment
# rather than adding a `WrapperSpec`.
WRAPPER_NAMES = [name for name in gym.wrappers.__all__ if name != "vector"]


def recorded_argument_names(wrapper: type) -> set[str]:
    """The keyword names that `wrapper.__init__` passes on to `RecordConstructorArgs.__init__`."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(wrapper.__init__)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "__init__"
            and "RecordConstructorArgs" in ast.unparse(node.func)
        ):
            return {
                keyword.arg
                for keyword in node.keywords
                if keyword.arg != "_disable_deepcopy"
            }

    raise AssertionError(
        f"{wrapper.__name__}.__init__ does not call RecordConstructorArgs.__init__"
    )


def constructor_argument_names(wrapper: type) -> set[str]:
    """The named parameters of `wrapper.__init__`, other than `self` and the wrapped environment."""
    parameters = inspect.signature(wrapper.__init__).parameters
    return {
        name
        for name, parameter in parameters.items()
        if name not in ("self", "env")
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }


@pytest.mark.parametrize(
    "wrapper_name",
    [
        (
            pytest.param(
                name,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Array API namespaces are modules, which RecordConstructorArgs cannot deepcopy",
                ),
            )
            if name == "ArrayConversion"
            else name
        )
        for name in WRAPPER_NAMES
    ],
)
def test_wrappers_record_every_constructor_argument(wrapper_name: str):
    """Every constructor argument of a wrapper is passed on to `RecordConstructorArgs`.

    `Wrapper.spec` puts the saved arguments into a `WrapperSpec`, and `gym.make`
    splats them back into the constructor when an environment is rebuilt from its
    spec. An argument that is never recorded is dropped by that round trip, and an
    argument recorded under a name the constructor does not take raises `TypeError`.
    """
    try:
        wrapper = getattr(gym.wrappers, wrapper_name)
    except gym.error.DependencyNotInstalled as e:
        pytest.skip(str(e))

    # JaxToNumpy, JaxToTorch and NumpyToTorch never call RecordConstructorArgs,
    # so there is nothing here to compare against their signatures. Not
    # recording an argument and not recording at all are separate gaps.
    if not issubclass(wrapper, RecordConstructorArgs):
        pytest.skip(f"{wrapper_name} does not record its constructor arguments at all")

    assert recorded_argument_names(wrapper) == constructor_argument_names(wrapper)


def test_env_spec_roundtrip_keeps_wrapper_arguments():
    """An environment rebuilt from its spec is wrapped with the arguments it was given."""
    env = gym.wrappers.DiscretizeAction(
        gym.make("Pendulum-v1"), bins=2, multidiscrete=True
    )
    assert gym.make(env.spec).action_space == env.action_space

    env = gym.wrappers.RecordEpisodeStatistics(
        gym.make("CartPole-v1"), buffer_length=17, stats_key="my_stats"
    )
    rebuilt = gym.make(env.spec)
    assert rebuilt._stats_key == env._stats_key
    assert rebuilt.time_queue.maxlen == env.time_queue.maxlen
