import pickle
import re
from copy import deepcopy

import numpy as np
import pytest

from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    GraphInstance,
    MultiBinary,
    MultiDiscrete,
    Tuple,
)
from gymnasium.utils.env_checker import data_equivalence


@pytest.mark.parametrize("feature_type", ["dict", "tuple", "nested"])
@pytest.mark.parametrize("feature_location", ["nodes", "edges"])
@pytest.mark.parametrize(
    "feature",
    [Box(0, 1, (3,)), Discrete(1000), MultiBinary(32), MultiDiscrete([1000, 1000])],
)
def test_composite_feature_sampling_advances(feature_type, feature_location, feature):
    feature = deepcopy(feature)
    if feature_type == "dict":
        feature = Dict(value=feature)
    elif feature_type == "tuple":
        feature = Tuple((feature,))
    else:
        feature = Dict(value=Tuple((Dict(value=feature),)))

    if feature_location == "nodes":
        space = Graph(feature, None, seed=42)
        sample_kwargs = {"num_nodes": 4}
    else:
        space = Graph(Discrete(5), feature, seed=42)
        sample_kwargs = {"num_nodes": 4, "num_edges": 3}

    samples = [space.sample(**sample_kwargs) for _ in range(3)]
    assert all(sample in space for sample in samples)
    for previous, current in zip(samples, samples[1:], strict=False):
        assert not data_equivalence(
            getattr(previous, feature_location), getattr(current, feature_location)
        )

    space.seed(42)
    assert all(
        data_equivalence(sample, space.sample(**sample_kwargs)) for sample in samples
    )


@pytest.mark.parametrize("feature_type", ["dict", "tuple", "nested"])
def test_composite_feature_sampling_lifecycle(feature_type):
    feature = Box(0, 1, (3,))
    if feature_type == "dict":
        feature = Dict(value=feature)
    elif feature_type == "tuple":
        feature = Tuple((feature,))
    else:
        feature = Dict(value=Tuple((Dict(value=feature),)))

    space = Graph(feature, deepcopy(feature), seed=42)
    seeds = space.seed(42)
    first = space.sample(num_nodes=4, num_edges=3)
    space.seed(seeds)
    assert data_equivalence(first, space.sample(num_nodes=4, num_edges=3))
    restored = pickle.loads(pickle.dumps(space))
    for num_nodes, num_edges in [(2, 1), (5, 4), (3, 0)]:
        assert data_equivalence(
            space.sample(num_nodes=num_nodes, num_edges=num_edges),
            restored.sample(num_nodes=num_nodes, num_edges=num_edges),
        )


@pytest.mark.parametrize("mask_type", ["mask", "probability"])
def test_composite_feature_sampling_masks(mask_type):
    feature = Dict(value=Tuple((Discrete(3),)))
    space = Graph(feature, deepcopy(feature), seed=42)
    dtype = np.int8 if mask_type == "mask" else np.float64
    node_mask = {"value": ((np.array([0, 1, 0], dtype=dtype),) * 4,)}
    edge_mask = {"value": ((np.array([0, 0, 1], dtype=dtype),) * 3,)}
    for _ in range(3):
        sample = space.sample(
            **{mask_type: (node_mask, edge_mask)}, num_nodes=4, num_edges=3
        )
        assert sample in space
        assert np.all(sample.nodes["value"][0] == 1)
        assert np.all(sample.edges["value"][0] == 2)


def test_node_space_sample():
    space = Graph(node_space=Discrete(3), edge_space=None)
    space.seed(0)

    sample = space.sample(
        mask=(tuple(np.array([0, 1, 0], dtype=np.int8) for _ in range(5)), None),
        num_nodes=5,
    )
    assert sample in space
    assert np.all(sample.nodes == 1)

    sample = space.sample(
        (
            (np.array([1, 0, 0], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)),
            None,
        ),
        num_nodes=2,
    )
    assert sample in space
    assert np.all(sample.nodes == np.array([0, 1]))

    with pytest.warns(
        UserWarning,
        match=re.escape("The number of edges is set (5) but the edge space is None."),
    ):
        sample = space.sample(num_edges=5)
        assert sample in space

    # Change the node_space or edge_space to a non-Box or discrete space.
    # This should not happen, test is primarily to increase coverage.
    with pytest.raises(
        TypeError,
        match=re.escape(
            "The space provided to `batch_space` is not a gymnasium Space instance, type: <class 'str'>, abc"
        ),
    ):
        space.node_space = "abc"
        space.sample()


def test_edge_space_sample():
    space = Graph(node_space=Discrete(3), edge_space=Discrete(3))
    space.seed(0)
    # When num_nodes>1 then num_edges is set to 0
    assert space.sample(num_nodes=1).edges is None
    assert 0 <= len(space.sample(num_edges=3).edges) < 6

    sample = space.sample(mask=(None, np.array([0, 1, 0], dtype=np.int8)))
    assert np.all(sample.edges == 1) or sample.edges is None

    sample = space.sample(
        mask=(
            None,
            (
                np.array([1, 0, 0], dtype=np.int8),
                np.array([0, 1, 0], dtype=np.int8),
                np.array([0, 0, 1], dtype=np.int8),
            ),
        ),
        num_edges=3,
    )
    assert np.all(sample.edges == np.array([0, 1, 2]))

    with pytest.raises(
        AssertionError,
        match="Expects the number of edges to be greater than 0, actual value: -1",
    ):
        space.sample(num_edges=-1)

    space = Graph(node_space=Discrete(3), edge_space=None)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "\x1b[33mWARN: The number of edges is set (5) but the edge space is None.\x1b[0m"
        ),
    ):
        sample = space.sample(num_edges=5)
    assert sample.edges is None


@pytest.mark.parametrize(
    "sample",
    [
        "abc",
        GraphInstance(
            nodes=None, edges=np.array([0, 1]), edge_links=np.array([[0, 1], [1, 0]])
        ),
        GraphInstance(
            nodes=np.array([10, 1, 0]),
            edges=np.array([0, 1]),
            edge_links=np.array([[0, 1], [1, 0]]),
        ),
        GraphInstance(
            nodes=np.array([0, 1]), edges=None, edge_links=np.array([[0, 1], [1, 0]])
        ),
        GraphInstance(nodes=np.array([0, 1]), edges=np.array([0, 1]), edge_links=None),
        GraphInstance(
            nodes=np.array([1, 2]),
            edges=np.array([10, 1]),
            edge_links=np.array([[0, 1], [1, 0]]),
        ),
        GraphInstance(
            nodes=np.array([1, 2]),
            edges=np.array([0, 1]),
            edge_links=np.array([[0.5, 1.0], [2.0, 1.0]]),
        ),
        GraphInstance(
            nodes=np.array([1, 2]), edges=np.array([10, 1]), edge_links=np.array([0, 1])
        ),
        GraphInstance(
            nodes=np.array([1, 2]),
            edges=np.array([0, 1]),
            edge_links=np.array([[[0], [1]], [[0], [0]]]),
        ),
        GraphInstance(
            nodes=np.array([1, 2]),
            edges=np.array([0, 1]),
            edge_links=np.array([[10, 1], [0, 0]]),
        ),
        GraphInstance(
            nodes=np.array([1, 2]),
            edges=np.array([0, 1]),
            edge_links=np.array([[-10, 1], [0, 0]]),
        ),
    ],
)
def test_not_contains(sample):
    space = Graph(node_space=Discrete(2), edge_space=Discrete(2))
    assert sample not in space


def test_probability_node_sampling():
    """
    Test the probability parameter for node sampling.
    Ensures nodes are sampled according to the given probability distribution.
    """
    space = Graph(node_space=Discrete(3), edge_space=None)
    space.seed(42)

    # Define a probability distribution for nodes
    probability = np.array([0.7, 0.2, 0.1], dtype=np.float64)
    num_samples = 1000

    # Collect samples with the given probability
    samples = [
        space.sample(probability=((probability,), None), num_nodes=1).nodes[0]
        for _ in range(num_samples)
    ]

    # Check the empirical distribution of the samples
    counts = np.bincount(samples, minlength=3)
    empirical_distribution = counts / num_samples

    assert np.allclose(empirical_distribution, probability, atol=0.05), (
        f"Empirical distribution {empirical_distribution} does not match expected probability {probability}"
    )


def test_probability_edge_sampling():
    """
    Test the probability parameter for edge sampling.
    Ensures edges are sampled according to the given probability distribution.
    """
    space = Graph(node_space=Discrete(3), edge_space=Discrete(3))
    space.seed(42)

    # Define a probability distribution for edges
    probability = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    num_samples = 1000

    # Collect samples with the given probability
    samples = [
        space.sample(probability=(None, (probability,)), num_edges=1).edges[0]
        for _ in range(num_samples)
    ]

    # Check the empirical distribution of the samples
    counts = np.bincount(samples, minlength=3)
    empirical_distribution = counts / num_samples

    assert np.allclose(empirical_distribution, probability, atol=0.05), (
        f"Empirical distribution {empirical_distribution} does not match expected probability {probability}"
    )


def test_probability_node_and_edge_sampling():
    """
    Test the probability parameter for both node and edge sampling.
    Ensures nodes and edges are sampled correctly according to their respective probability distributions.
    """
    space = Graph(node_space=Discrete(3), edge_space=Discrete(3))
    space.seed(42)

    # Define probability distributions for nodes and edges
    node_probability = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    edge_probability = np.array([0.4, 0.4, 0.2], dtype=np.float64)
    num_samples = 1000

    # Collect samples with the given probabilities
    node_samples = []
    edge_samples = []
    for _ in range(num_samples):
        sample = space.sample(
            probability=((node_probability,), (edge_probability,)),
            num_nodes=1,
            num_edges=1,
        )
        node_samples.append(sample.nodes[0])
        edge_samples.append(sample.edges[0])

    # Check the empirical distributions of the samples
    node_counts = np.bincount(node_samples, minlength=3)
    edge_counts = np.bincount(edge_samples, minlength=3)

    node_empirical_distribution = node_counts / num_samples
    edge_empirical_distribution = edge_counts / num_samples

    assert np.allclose(node_empirical_distribution, node_probability, atol=0.05), (
        f"Node empirical distribution {node_empirical_distribution} does not match expected probability {node_probability}"
    )

    assert np.allclose(edge_empirical_distribution, edge_probability, atol=0.05), (
        f"Edge empirical distribution {edge_empirical_distribution} does not match expected probability {edge_probability}"
    )
