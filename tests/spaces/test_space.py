from functools import partial

import pytest

from gymnasium.spaces import MultiBinary, MultiDiscrete, utils
from tests.spaces.utils import TESTING_CUSTOM_SPACE


@pytest.mark.parametrize(
    "func",
    [
        TESTING_CUSTOM_SPACE.sample,
        partial(TESTING_CUSTOM_SPACE.contains, None),
        partial(utils.flatdim, TESTING_CUSTOM_SPACE),
        partial(utils.flatten, TESTING_CUSTOM_SPACE, None),
        partial(utils.flatten_space, TESTING_CUSTOM_SPACE),
        partial(utils.unflatten, TESTING_CUSTOM_SPACE, None),
    ],
)
def test_not_implemented_errors(func):
    with pytest.raises(NotImplementedError):
        func()


@pytest.mark.parametrize(
    "space", [MultiBinary((2, 2)), MultiDiscrete([[2, 2], [2, 2]])]
)
@pytest.mark.parametrize("sample", [[[0], [0, 1]], ((0,), (0, 1))])
def test_contains_ragged_sequence(space, sample):
    assert space.contains(sample) is False
    assert (sample in space) is False


@pytest.mark.parametrize(
    "space", [MultiBinary((2, 2)), MultiDiscrete([[2, 2], [2, 2]])]
)
@pytest.mark.parametrize("sample", [[[0, 1], [1, 0]], ((0, 1), (1, 0))])
def test_contains_rectangular_sequence(space, sample):
    assert space.contains(sample) is True
