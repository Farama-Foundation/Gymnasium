"""Tests `gymnasium.experimental.vector.utils.shared_memory functions."""

import multiprocessing as mp

import pytest

from gymnasium.error import CustomSpaceError
from gymnasium.experimental.vector.utils import (
    create_shared_memory,
    read_from_shared_memory,
    write_to_shared_memory,
)
from gymnasium.utils.env_checker import data_equivalence
from tests.spaces.utils import TESTING_SPACES, TESTING_SPACES_IDS, CUSTOM_SPACES, CUSTOM_SPACES_IDS


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
@pytest.mark.parametrize("num", [1, 8])
@pytest.mark.parametrize(
    "ctx", [None, "fork", "spawn"], ids=["default", "fork", "spawn"]
)
def test_shared_memory_create_read_write(space, num, ctx):
    """Test the shared memory functions, create, read and write for all of the testing spaces."""
    ctx = mp if ctx is None else mp.get_context(ctx)
    samples = [space.sample() for _ in range(num)]

    try:
        shared_memory = create_shared_memory(space, n=num, ctx=ctx)
    except TypeError:
        return

    for i, sample in enumerate(samples):
        write_to_shared_memory(space, i, sample, shared_memory)

    read_samples = read_from_shared_memory(space, shared_memory, n=num)
    for read_sample, sample in zip(read_samples, samples):
        data_equivalence(read_sample, sample)


@pytest.mark.parametrize("space", CUSTOM_SPACES, ids=CUSTOM_SPACES_IDS)
def test_shared_memory_custom_space(space):
    with pytest.raises(CustomSpaceError):
        create_shared_memory(space)
