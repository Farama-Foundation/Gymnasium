"""Test suite for import wrappers."""

import re

import pytest

import gymnasium
import gymnasium.experimental.wrappers as wrappers
from gymnasium.experimental.wrappers import __all__


def test_import_wrappers():
    """Test that all wrappers can be imported."""
    # Test that a deprecated wrapper raises a DeprecatedWrapper
    with pytest.raises(
        wrappers.DeprecatedWrapper,
        match=re.escape("'NormalizeRewardV0' is now deprecated"),
    ):
        getattr(wrappers, "NormalizeRewardV0")

    # Test that an invalid version raises an AttributeError
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "module 'gymnasium.experimental.wrappers' has no attribute 'ClipRewardVT', did you mean"
        ),
    ):
        getattr(wrappers, "ClipRewardVT")

    with pytest.raises(
        AttributeError,
        match=re.escape(
            "module 'gymnasium.experimental.wrappers' has no attribute 'ClipRewardV99', did you mean"
        ),
    ):
        getattr(wrappers, "ClipRewardV99")

    # Test that an invalid wrapper raises an AttributeError
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "module 'gymnasium.experimental.wrappers' has no attribute 'NonexistentWrapper'"
        ),
    ):
        getattr(wrappers, "NonexistentWrapper")


@pytest.mark.parametrize("wrapper_name", __all__)
def test_all_wrappers_shortened(wrapper_name):
    """Check that each element of the `__all__` wrappers can be loaded, provided dependencies are installed."""
    try:
        assert getattr(gymnasium.experimental.wrappers, wrapper_name) is not None
    except gymnasium.error.DependencyNotInstalled as e:
        pytest.skip(str(e))


def test_wrapper_vector():
    assert gymnasium.experimental.wrappers.vector is not None
