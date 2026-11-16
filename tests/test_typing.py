"""Static type checks for Gymnasium APIs."""

from typing import TYPE_CHECKING, assert_type

import numpy as np

from gymnasium.spaces import Discrete, Sequence

if TYPE_CHECKING:
    sequence = Sequence(Discrete(3))
    for length in (
        None,
        3,
        np.int64(3),
        np.uint64(3),
        np.array([2, 3], dtype=np.int64),
    ):
        sequence.sample(mask=(length, None))
        sequence.sample(probability=(length, None))
        assert_type(sequence.generate_sample_length(length, "mask"), int | np.integer)
