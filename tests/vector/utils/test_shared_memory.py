"""Tests `gymnasium.vector.utils.shared_memory functions."""

import multiprocessing as mp
import re

import pytest

from gymnasium import Space
from gymnasium.error import CustomSpaceError
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector.utils import (
    batch_space,
    create_shared_memory,
    iterate,
    read_from_shared_memory,
    write_to_shared_memory,
)
from tests.spaces.utils import TESTING_SPACES, TESTING_SPACES_IDS


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
@pytest.mark.parametrize("num", [1, 8])
@pytest.mark.parametrize(
    "ctx", [None, "fork", "spawn"], ids=["default", "fork", "spawn"]
)
def test_shared_memory_create_read_write(space, num, ctx):
    """Test the shared memory functions, create, read and write for all testing spaces."""
    try:
        _ctx = mp.get_context(ctx)
    except ValueError:
        pytest.skip(
            f"Multiprocessing start method {ctx} not available on this platform."
        )
    ctx = _ctx

    batched_space = batch_space(space, n=num)
    try:
        shared_memory = create_shared_memory(space, n=num, ctx=ctx)
    except TypeError as err:
        assert (
            "has a dynamic shape so its not possible to make a static shared memory."
            in str(err)
        )
        pytest.skip("Skipping space with dynamic shape")

    samples = [space.sample() for _ in range(num)]
    for i, sample in enumerate(samples):
        write_to_shared_memory(space, i, sample, shared_memory)

    read_samples = read_from_shared_memory(space, shared_memory, n=num)
    assert read_samples in batched_space
    for read_sample, sample in zip(iterate(batched_space, read_samples), samples):
        assert data_equivalence(read_sample, sample)


def test_custom_space():
    """Test using custom spaces for shared memory functions."""
    with pytest.raises(
        CustomSpaceError,
        match=re.escape(
            "Space of type `<class 'gymnasium.spaces.space.Space'>` doesn't have an registered `create_shared_memory` function. Register `<class 'gymnasium.spaces.space.Space'>` for `create_shared_memory` to support it."
        ),
    ):
        create_shared_memory(Space())

    with pytest.raises(
        CustomSpaceError,
        match=re.escape(
            "Space of type `<class 'gymnasium.spaces.space.Space'>` doesn't have an registered `read_from_shared_memory` function. Register `<class 'gymnasium.spaces.space.Space'>` for `read_from_shared_memory` to support it."
        ),
    ):
        read_from_shared_memory(Space(), None, 1)

    with pytest.raises(
        CustomSpaceError,
        match=re.escape(
            "Space of type `<class 'gymnasium.spaces.space.Space'>` doesn't have an registered `write_to_shared_memory` function. Register `<class 'gymnasium.spaces.space.Space'>` for `write_to_shared_memory` to support it."
        ),
    ):
        write_to_shared_memory(Space(), 1, None, None)


def test_non_space():
    """Test the use of non-space types on the shared memory functions."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "The space provided to `create_shared_memory` is not a gymnasium Space instance, type: <class 'str'>, space"
        ),
    ):
        create_shared_memory("space")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The space provided to `read_from_shared_memory` is not a gymnasium Space instance, type: <class 'str'>, space"
        ),
    ):
        read_from_shared_memory("space", None, 1)

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The space provided to `write_to_shared_memory` is not a gymnasium Space instance, type: <class 'str'>, space"
        ),
    ):
        write_to_shared_memory("space", 1, None, None)
