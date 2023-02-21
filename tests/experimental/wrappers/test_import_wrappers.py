"""Test suite for import wrappers."""

import re

import pytest

import gymnasium.experimental.wrappers as wrappers


def test_import_wrappers():
    """Test that all wrappers can be imported."""
    # Test that a deprecated wrapper raises a DeprecatedWrapper
    with pytest.raises(
        wrappers.DeprecatedWrapper,
        match=re.escape("NormalizeRewardV0 is now deprecated"),
    ):
        getattr(wrappers, "NormalizeRewardV0")

    # Test that an invalid version raises an InvalidVersionWrapper
    with pytest.raises(
        wrappers.InvalidVersionWrapper,
        match=re.escape("ClipRewardVT is not a valid version number"),
    ):
        getattr(wrappers, "ClipRewardVT")

    # Test that an invalid wrapper raises an AttributeError
    with pytest.raises(
        AttributeError, match=re.escape("cannot import name 'NonexistentWrapper'")
    ):
        getattr(wrappers, "NonexistentWrapper")
