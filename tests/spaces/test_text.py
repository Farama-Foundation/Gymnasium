import os
import re
import subprocess
import sys

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


def test_charset_ordering():
    # set-based charsets have no inherent order so they are sorted
    space = Text(5, charset=frozenset("dcba"))
    assert space.character_list == ("a", "b", "c", "d")
    assert [space.character_index(c) for c in "abcd"] == [0, 1, 2, 3]

    # string charsets keep the given order, dropping duplicate characters
    space = Text(5, charset="dcbad")
    assert space.character_list == ("d", "c", "b", "a")
    assert [space.character_index(c) for c in "dcba"] == [0, 1, 2, 3]


def test_deterministic_across_hash_seeds():
    """Seeded samples and flatten encodings must not depend on PYTHONHASHSEED.

    The default charset is a frozenset whose iteration order changes with hash
    randomization, so the character ordering must be normalized.
    """
    code = (
        "from gymnasium.spaces import Text\n"
        "from gymnasium.spaces.utils import flatten\n"
        "space = Text(5, seed=42)\n"
        "print(space.sample())\n"
        "print(flatten(Text(5), 'abc').tolist())\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in (1, 2)
    }
    assert len(outputs) == 1, f"Outputs differ across hash seeds: {outputs}"
