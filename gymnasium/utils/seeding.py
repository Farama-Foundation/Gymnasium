"""Set of random number generator functions: seeding, generator, hashing seeds."""

from __future__ import annotations

import numpy as np

from gymnasium import error


def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Returns a NumPy random number generator (RNG) along with seed value from the inputted seed.

    If ``seed`` is ``None`` then a **random** seed will be generated as the RNG's initial seed.
    This randomly selected seed is returned as the second value of the tuple.

    .. py:currentmodule:: gymnasium.Env

    This function is called in :meth:`reset` to reset an environment's initial RNG.

    Args:
        seed: The seed used to create the generator

    Returns:
        A NumPy-based Random Number Generator and generator seed

    Raises:
        Error: Seed must be a non-negative integer
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
