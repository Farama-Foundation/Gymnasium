"""Utility functions for testing the vector utility functions."""

import numpy as np


def is_rng_equal(rng_1: np.random.Generator, rng_2: np.random.Generator):
    """Asserts that two random number generates are equivalent."""
    return rng_1.bit_generator.state == rng_2.bit_generator.state


def type_equivalence(data_1, data_2):
    """Assert the type equivalences between two variables."""
    if type(data_1) is type(data_2):
        if isinstance(data_1, tuple):
            # assert len(data_1) == len(data_2), f'{len(data_1)}, {len(data_2)}, {data_1}, {data_2}'

            for o_1, o_2 in zip(data_1, data_2):
                assert type_equivalence(
                    o_1, o_2
                ), f"{type(o_1)}, {type(o_2)}, {o_1}, {o_2}"
        elif isinstance(data_1, dict):
            for key in data_1:
                assert type_equivalence(
                    data_1[key], data_2[key]
                ), f"{type(data_1[key])}, {type(data_2[key])}, {key}, {data_1[key]}, {data_2[key]}"

        return True
    assert False, f"{type(data_1)}, {type(data_2)}, {data_1}, {data_2}"
