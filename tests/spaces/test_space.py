from functools import partial

import pytest

from gymnasium.spaces import utils
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
