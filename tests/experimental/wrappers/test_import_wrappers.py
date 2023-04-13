"""Test suite for import wrappers."""

import re

import pytest

import gymnasium.experimental.wrappers as wrappers


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
