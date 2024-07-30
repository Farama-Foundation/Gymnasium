"""Test suite for import wrappers."""

import re

import pytest

import gymnasium
import gymnasium.wrappers as wrappers
from gymnasium.wrappers import __all__


def test_import_wrappers():
    """Test that all wrappers can be imported."""
    # Test that an invalid wrapper raises an AttributeError
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "module 'gymnasium.wrappers' has no attribute 'NonexistentWrapper'"
        ),
    ):
        getattr(wrappers, "NonexistentWrapper")


@pytest.mark.parametrize("wrapper_name", __all__)
def test_all_wrappers_shortened(wrapper_name):
    """Check that each element of the `__all__` wrappers can be loaded, provided dependencies are installed."""
    try:
        assert getattr(gymnasium.wrappers, wrapper_name) is not None
    except gymnasium.error.DependencyNotInstalled as e:
        pytest.skip(str(e))


def test_wrapper_vector():
    assert gymnasium.wrappers.vector is not None


@pytest.mark.parametrize(
    "wrapper_name",
    ("AutoResetWrapper", "FrameStack", "PixelObservationWrapper", "VectorListInfo"),
)
def test_renamed_wrappers(wrapper_name):
    with pytest.raises(
        AttributeError, match=f"{wrapper_name!r} has been renamed with"
    ) as err_message:
        getattr(wrappers, wrapper_name)

    new_wrapper_name = err_message.value.args[0][len(wrapper_name) + 35 : -1]
    if "vector." in new_wrapper_name:
        no_vector_wrapper_name = new_wrapper_name[len("vector.") :]
        assert getattr(gymnasium.wrappers.vector, no_vector_wrapper_name)
    else:
        assert getattr(gymnasium.wrappers, new_wrapper_name)
