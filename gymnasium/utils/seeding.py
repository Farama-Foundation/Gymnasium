"""Set of random number generator functions: seeding, generator, hashing seeds."""
from typing import Any, Optional, Tuple

import numpy as np

from gymnasium import error


def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, Any]:
    """Generates a random number generator from the seed and returns the Generator and seed.

    Args:
        seed: The seed used to create the generator

    Returns:
        The generator and resulting seed

    Raises:
        Error: Seed must be a non-negative integer or omitted
    """
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        if isinstance(seed, int) is False:
            raise error.Error(
                f"Seed must be a python integer, actual type: {type(seed)}"
            )
        else:
            raise error.Error(
                f"Seed must be greater or equal to zero, actual value: {seed}"
            )

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
    return rng, np_seed


RNG = RandomNumberGenerator = np.random.Generator
