"""Tests that all shortened imports for wrappers all work."""

import pytest

import gymnasium
from gymnasium.experimental.wrappers import (
    _wrapper_to_class,  # pyright: ignore[reportPrivateUsage]
)
from gymnasium.experimental.wrappers import __all__


def test_all_wrapper_shorten():
    """Test that all wrappers in `__all__` are contained within the `_wrapper_to_class` conversion."""
    all_wrappers = set(__all__)
    all_wrappers.remove("vector")
    assert all_wrappers == set(_wrapper_to_class.keys())


@pytest.mark.parametrize("wrapper_name", __all__)
def test_all_wrappers_shortened(wrapper_name):
    """Check that each element of the `__all__` wrappers can be loaded, provided dependencies are installed."""
    if wrapper_name != "vector":
        try:
            assert getattr(gymnasium.experimental.wrappers, wrapper_name) is not None
        except gymnasium.error.DependencyNotInstalled as e:
            pytest.skip(str(e))
