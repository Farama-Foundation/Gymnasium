import numpy as np
import pytest

from gymnasium.spaces import MultiBinary


def test_sample():
    space = MultiBinary(4)

    sample = space.sample(mask=np.array([0, 0, 1, 1], dtype=np.int8))
    assert np.all(sample == [0, 0, 1, 1])

    sample = space.sample(mask=np.array([0, 1, 2, 2], dtype=np.int8))
    assert sample[0] == 0 and sample[1] == 1
    assert sample[2] == 0 or sample[2] == 1
    assert sample[3] == 0 or sample[3] == 1

    space = MultiBinary(np.array([2, 3]))
    sample = space.sample(mask=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int8))
    assert np.all(sample == [[0, 0, 0], [1, 1, 1]]), sample


def test_sample_probabilities():
    # Test sampling with probabilities
    space = MultiBinary(4)
    probabilities = np.array([0, 1, 0.5, 0.25], dtype=np.float64)

    samples = [space.sample(probability=probabilities) for _ in range(10000)]
    assert all(sample in space for sample in samples)
    samples = np.array(samples)

    # Check empirical probabilities
    for i in range(4):
        counts = np.sum(samples[:, i]) / len(samples)
        np.testing.assert_allclose(counts, probabilities[i], atol=0.05)


def test_sample_zero_probability_at_zero_uniform_draw():
    """A valid zero uniform draw must not select probability-zero elements."""
    draws = np.array([[0.0, 0.0, 0.25], [0.0, 0.999, 0.75]])

    class FixedGenerator(np.random.Generator):
        """Return deterministic values within Generator.random's [0, 1) range."""

        def random(self, size=None, dtype=np.float64, out=None):
            """Return the prescribed draws without changing their shape."""
            assert size == draws.shape
            return draws.copy()

    space = MultiBinary((2, 3), seed=FixedGenerator(np.random.PCG64(0)))
    probabilities = np.array([[0.0, 1.0, 0.5], [0.0, 1.0, 0.5]])
    sample = space.sample(probability=probabilities)
    np.testing.assert_array_equal(sample, [[0, 1, 1], [0, 1, 0]])
    assert sample.dtype == space.dtype
    assert sample in space


@pytest.mark.parametrize("n", [np.int64(4), np.int32(4), np.uint8(4), np.prod((2, 2))])
def test_numpy_integer_n(n):
    space = MultiBinary(n)

    assert space.n == 4 and isinstance(space.n, int)
    assert space.shape == (4,)
    assert space == MultiBinary(4)
    assert space.sample() in space


def test_non_positive_numpy_integer_n():
    with pytest.raises(ValueError, match="n \\(counts\\) have to be positive"):
        MultiBinary(np.int64(0))
