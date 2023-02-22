"""Utility functions for testing the vector utility functions."""
import numpy as np


def is_rng_equal(rng_1: np.random.Generator, rng_2: np.random.Generator):
    """Asserts that two random number generates are equivalent."""
    return rng_1.bit_generator.state == rng_2.bit_generator.state
