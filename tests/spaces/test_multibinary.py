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
