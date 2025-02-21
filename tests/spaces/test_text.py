import re

import numpy as np
import pytest

from gymnasium.spaces import Text


def test_sample_mask():
    space = Text(min_length=1, max_length=5)

    # Test the sample length
    sample = space.sample(mask=(3, None))
    assert sample in space
    assert len(sample) == 3

    sample = space.sample(mask=None)
    assert sample in space
    assert 1 <= len(sample) <= 5

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Trying to sample with a minimum length > 0 (actual minimum length=1) but the character mask is all zero meaning that no character could be sampled."
        ),
    ):
        space.sample(mask=(3, np.zeros(len(space.character_set), dtype=np.int8)))

    space = Text(min_length=0, max_length=5)
    sample = space.sample(
        mask=(None, np.zeros(len(space.character_set), dtype=np.int8))
    )
    assert sample in space
    assert sample == ""

    sample = space.sample(mask=(0, None))
    assert sample in space
    assert sample == ""

    # Test the sample characters
    space = Text(max_length=5, charset="abcd")

    sample = space.sample(mask=(3, np.array([0, 1, 0, 0], dtype=np.int8)))
    assert sample in space
    assert sample == "bbb"


def test_sample_probability():
    space = Text(min_length=1, max_length=5)

    # Test the sample length
    sample = space.sample(probability=(3, None))
    assert sample in space
    assert len(sample) == 3

    sample = space.sample(probability=None)
    assert sample in space
    assert 1 <= len(sample) <= 5

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expects the sum of the probability mask to be 1, actual sum: 0.0"
        ),
    ):
        space.sample(
            probability=(3, np.zeros(len(space.character_set), dtype=np.float64))
        )

    # Test the sample characters
    space = Text(max_length=5, charset="abcd")

    sample = space.sample(probability=(3, np.array([0, 1, 0, 0], dtype=np.float64)))
    assert sample in space
    assert sample == "bbb"

    sample = space.sample(probability=(2, np.array([0.5, 0.5, 0, 0], dtype=np.float64)))
    assert sample in space
    assert sample in ["aa", "bb", "ab", "ba"]
