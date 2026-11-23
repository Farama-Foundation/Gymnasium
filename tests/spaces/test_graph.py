import json
import re

import numpy as np
import pytest

from gymnasium.spaces import Box, Discrete, Graph, GraphInstance


@pytest.mark.parametrize(
    "node_space", [Discrete(3), Box(-1, 1, shape=(2,), dtype=np.float32)]
)
@pytest.mark.parametrize("num_edges", [0, 1, 3])
def test_graph_jsonable_edge_shapes(node_space: Box | Discrete, num_edges: int):
    """Graph JSON round-trips preserve edge-link shapes, including empty edges."""
    space = Graph(node_space=node_space, edge_space=Discrete(3), seed=0)
    sample = GraphInstance(
        nodes=space.sample(num_nodes=3, num_edges=1).nodes,
        edges=np.arange(num_edges, dtype=np.int64) % 3,
        edge_links=np.tile(np.array([[0, 1]], dtype=np.int32), (num_edges, 1)),
    )
    assert sample in space

    encoded = json.loads(json.dumps(space.to_jsonable([sample])))
    restored = space.from_jsonable(encoded)[0]

    assert restored in space
    np.testing.assert_array_equal(restored.nodes, sample.nodes)
    np.testing.assert_array_equal(restored.edges, sample.edges)
    np.testing.assert_array_equal(restored.edge_links, sample.edge_links)
    assert restored.edge_links.shape == (num_edges, 2)
    assert restored.edge_links.dtype == np.int32


def test_graph_jsonable_mixed_empty_edges():
    """An empty edge list in a batch does not change other graph samples."""
    space = Graph(node_space=Discrete(3), edge_space=Discrete(3))
    samples = [
        GraphInstance(
            nodes=np.array([0, 1]),
            edges=np.arange(count, dtype=np.int64),
            edge_links=np.tile(np.array([[0, 1]], dtype=np.int32), (count, 1)),
        )
        for count in [0, 1]
    ]
    assert all(sample in space for sample in samples)

    encoded = json.loads(json.dumps(space.to_jsonable(samples)))
    restored = space.from_jsonable(encoded)

    assert len(restored) == len(samples)
    for original, decoded in zip(samples, restored, strict=True):
        assert decoded in space
        np.testing.assert_array_equal(decoded.edge_links, original.edge_links)


def test_graph_jsonable_without_edge_space():
    """Graphs without an edge space retain absent edges and edge links."""
    space = Graph(node_space=Discrete(3), edge_space=None, seed=0)
    sample = space.sample(num_nodes=3)

    encoded = json.loads(json.dumps(space.to_jsonable([sample])))
    restored = space.from_jsonable(encoded)[0]

    assert restored in space
    np.testing.assert_array_equal(restored.nodes, sample.nodes)
    assert restored.edges is None
    assert restored.edge_links is None


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
