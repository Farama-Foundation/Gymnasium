from functools import partial

import pytest

from gymnasium.spaces import utils
from tests.spaces.utils import TESTING_BASE_SPACE


@pytest.mark.parametrize(
    "func",
    [
        TESTING_BASE_SPACE.sample,
        partial(TESTING_BASE_SPACE.contains, None),
        partial(utils.flatdim, TESTING_BASE_SPACE),
        partial(utils.flatten, TESTING_BASE_SPACE, None),
        partial(utils.flatten_space, TESTING_BASE_SPACE),
        partial(utils.unflatten, TESTING_BASE_SPACE, None),
    ],
)
def test_not_implemented_errors(func):
    with pytest.raises(NotImplementedError):
        func()
