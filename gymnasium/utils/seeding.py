"""Random number generator functions to make new np.random.generator: np_random and variables: RNG, RandomNumberGenerator."""

import numpy as np

from gymnasium._random import np_random


RNG = RandomNumberGenerator = np.random.Generator

__all__ = ["RNG", "RandomNumberGenerator", "np_random"]
