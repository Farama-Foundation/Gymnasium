from copy import deepcopy

import numpy as np

from gymnasium.spaces import Discrete


def test_space_legacy_pickling():
    """Test the legacy pickle of Discrete that is missing the `start` parameter."""
    # Test that start is corrected passed
    space = Discrete(1, start=2)
    state = space.__dict__

    new_space = Discrete(1)
    new_space.__setstate__(state)
    assert new_space == space
    assert new_space.start == 2

    legacy_space = Discrete(1)
    legacy_state = deepcopy(legacy_space.__dict__)
    del legacy_state["start"]

    new_legacy_space = Discrete(2)
    new_legacy_space.__setstate__(legacy_state)
    assert new_legacy_space == legacy_space
    assert new_legacy_space.start == 0


def test_sample_mask():
    """Test that the mask parameter of the sample function works as expected."""
    space = Discrete(4, start=2)
    assert 2 <= space.sample() < 6
    assert space.sample(mask=np.array([0, 1, 0, 0], dtype=np.int8)) == 3
    assert space.sample(mask=np.array([0, 0, 0, 0], dtype=np.int8)) == 2
    assert space.sample(mask=np.array([0, 1, 0, 1], dtype=np.int8)) in [3, 5]


def test_probability_mask():
    """Test that the probability parameter of the sample function works as expected."""
    space = Discrete(4, start=2)
    assert space.sample(probability=np.array([0, 1, 0, 0], dtype=np.float64)) == 3
    assert space.sample(mask=np.array([0, 0.5, 0, 0.5], dtype=np.float64)) in [3, 5]
    assert space.sample(mask=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)) in [
        2,
        3,
        4,
        5,
    ]


def test_invalid_probability_mask():
    """Test that invalid activities raise the correct exception."""
    space = Discrete(4, start=2)
    try:
        space.sample(
            mask=np.array([0, 1, 0, 0], dtype=np.int8),
            probability=np.array([0, 1, 0, 0], dtype=np.float64),
        )
    except AssertionError as e:
        assert (
            str(e) == "Either mask or probability can be provided, not both"
        ), f"unexpected error message: {e}"
    else:
        assert False, "Expected AssertionError not raised"

    try:
        space.sample(probability=np.array([0, 1, 0, 0], dtype=np.int8))
    except AssertionError as e:
        assert (
            str(e)
            == "The expected dtype of the probability mask is np.float64, actual dtype: int8"
        ), f"unexpected error message: {e}"
    else:
        assert False, "Expected AssertionError not raised"

    try:
        space.sample(probability=np.array([-0.5, 1, 0.5, 0], dtype=np.float64))
    except AssertionError as e:
        assert (
            str(e)
            == "All values of a mask should be 0, 1, or in between, actual values: [-0.5  1.   0.5  0. ]"
        ), f"unexpected error message: {e}"
    else:
        assert False, "Expected AssertionError not raised"

    try:
        space.sample(probability=np.array([0.2, 0.3, 0.4, 0.2], dtype=np.float64))
    except AssertionError as e:
        assert (
            str(e)
            == "The sum of all values of the probability mask should be 1, actual sum: 1.1"
        ), f"unexpected error message: {e}"
    else:
        assert False, "Expected AssertionError not raised"

    try:
        space.sample(probability=np.array([0, 0, 0, 0], dtype=np.float64))
    except AssertionError as e:
        assert (
            str(e)
            == "The sum of all values of the probability mask should be 1, actual sum: 0.0"
        ), f"unexpected error message: {e}"
    else:
        assert False, "Expected AssertionError not raised"
