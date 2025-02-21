import numpy as np

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
