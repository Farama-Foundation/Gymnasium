import json
import pickle
import re
from copy import deepcopy

import numpy as np
import pytest

from gymnasium import spaces
from gymnasium.spaces import Discrete, Graph, GraphInstance
from gymnasium.spaces.utils import (
    flatten,
    flatten_space,
    is_space_dtype_shape_equiv,
    unflatten,
)
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector.utils import create_empty_array
from gymnasium.wrappers.utils import create_zero_array


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


@pytest.mark.parametrize(
    "counts,error",
    [
        ({"num_nodes": 3}, ValueError),
        ({"num_edges": 2}, ValueError),
        ({"num_nodes": True, "num_edges": 2}, TypeError),
        ({"num_nodes": 3, "num_edges": np.bool_(False)}, TypeError),
        ({"num_nodes": 3.0, "num_edges": 2}, TypeError),
        ({"num_nodes": 3, "num_edges": 2.0}, TypeError),
        ({"num_nodes": 0, "num_edges": 2}, ValueError),
        ({"num_nodes": 3, "num_edges": -1}, ValueError),
        ({"num_nodes": 2**31, "num_edges": 0}, ValueError),
    ],
)
def test_fixed_count_validation(counts, error):
    with pytest.raises(error):
        Graph(Discrete(3), Discrete(2), **counts)


@pytest.mark.parametrize(
    "feature",
    [
        spaces.Text(3),
        spaces.Sequence(Discrete(2)),
        spaces.OneOf([Discrete(2)]),
        Graph(Discrete(2), None),
        spaces.Dict({}),
        spaces.Tuple(()),
        spaces.Dict({"nested": spaces.Tuple((spaces.Text(3),))}),
        spaces.Space(),
    ],
)
@pytest.mark.parametrize("location", ["node_space", "edge_space"])
def test_fixed_unsupported_features(feature, location):
    kwargs = {"node_space": Discrete(3), "edge_space": Discrete(2), location: feature}
    with pytest.raises(TypeError, match=location):
        Graph(**kwargs, num_nodes=3, num_edges=2)


@pytest.mark.parametrize("edge_space", [None, Discrete(2)])
def test_fixed_zero_edges(edge_space):
    space = Graph(Discrete(3), edge_space, num_nodes=np.int64(3), num_edges=0)
    sample = space.sample(mask=(None, ()))
    assert sample in space
    assert sample.nodes.shape == (3,)
    assert sample.edges is sample.edge_links is None
    assert sample._replace(edges=np.empty(0, dtype=np.int64)) not in space
    with pytest.raises(ValueError, match="edge mask"):
        space.sample(mask=(None, (np.ones(2, dtype=np.int8),)))
    with pytest.raises(ValueError, match="zero"):
        Graph(Discrete(3), None, num_nodes=3, num_edges=1)


def test_fixed_sample_counts_and_rng():
    fixed = Graph(Discrete(5), Discrete(3), seed=42, num_nodes=3, num_edges=2)
    dynamic = Graph(Discrete(5), Discrete(3), seed=42)
    for _ in range(5):
        assert data_equivalence(
            fixed.sample(), dynamic.sample(num_nodes=3, num_edges=2)
        )
    reference = deepcopy(fixed)
    for counts in ({"num_nodes": 4}, {"num_edges": 3}, {"num_nodes": True}):
        with pytest.raises((ValueError, TypeError)):
            fixed.sample(**counts)
    assert data_equivalence(fixed.sample(num_nodes=3, num_edges=2), reference.sample())
    seeds = fixed.seed(21)
    expected = fixed.sample()
    fixed.seed(seeds)
    assert data_equivalence(fixed.sample(), expected)
    assert data_equivalence(pickle.loads(pickle.dumps(fixed)).sample(), fixed.sample())


@pytest.mark.parametrize("edge_space", [None, Discrete(2)])
@pytest.mark.parametrize(
    "node_space",
    [Discrete(3), spaces.Dict({"feature": spaces.Box(-1, 1, (2,))})],
)
def test_legacy_pickle_without_count_constraints(node_space, edge_space):
    """Graphs saved before fixed counts were added remain dynamic after loading."""
    legacy = Graph(deepcopy(node_space), deepcopy(edge_space), seed=42)
    reference = deepcopy(legacy)
    del legacy.num_nodes
    del legacy.num_edges

    restored = pickle.loads(pickle.dumps(legacy))
    assert restored.num_nodes is restored.num_edges is None
    assert restored == reference
    assert repr(restored) == repr(reference)
    assert data_equivalence(restored.sample(), reference.sample())
    assert restored.sample(num_nodes=3) in restored


@pytest.mark.parametrize("kind", ["mask", "probability"])
def test_fixed_per_feature_masks(kind):
    space = Graph(Discrete(3), Discrete(2), num_nodes=3, num_edges=2)
    dtype = np.int8 if kind == "mask" else np.float64
    nodes = tuple(np.eye(3, dtype=dtype))
    edges = tuple(np.eye(2, dtype=dtype))
    sample = space.sample(**{kind: (nodes, edges)})
    np.testing.assert_array_equal(sample.nodes, [0, 1, 2])
    np.testing.assert_array_equal(sample.edges, [0, 1])
    assert sample in space


def test_fixed_contains_checks_entire_layout():
    space = Graph(
        spaces.Dict({"feature": spaces.Box(-1, 1, (2,)), "label": Discrete(3)}),
        spaces.Tuple((Discrete(2), spaces.MultiBinary(2))),
        num_nodes=3,
        num_edges=2,
    )
    sample = space.sample()
    assert sample in space
    invalid_nodes = [
        {"feature": sample.nodes["feature"][:1], "label": sample.nodes["label"]},
        {"feature": sample.nodes["feature"]},
        {**sample.nodes, "label": np.full(3, 3)},
        {**sample.nodes, "feature": sample.nodes["feature"].astype(np.float64)},
    ]
    for nodes in invalid_nodes:
        assert sample._replace(nodes=nodes) not in space
    assert sample._replace(edges=(sample.edges[0][:1], sample.edges[1])) not in space
    for links in (
        None,
        np.zeros((1, 2), dtype=int),
        np.zeros((2, 2)),
        np.full((2, 2), -1),
        np.full((2, 2), 3),
    ):
        assert sample._replace(edge_links=links) not in space
    assert sample._replace(edge_links=sample.edge_links.astype(np.int64)) in space
    assert tuple(sample) not in space


@pytest.mark.parametrize("num_edges", [0, 2])
def test_fixed_helpers_keep_counts_and_structure(num_edges):
    space = Graph(
        spaces.Tuple(
            (spaces.Dict({"kind": Discrete(3, start=2)}), spaces.Box(1, 2, (2,)))
        ),
        spaces.Dict({"weight": spaces.Box(1, 2, ())}),
        num_nodes=3,
        num_edges=num_edges,
    )
    sample = space.sample()
    recovered = space.from_jsonable(
        json.loads(json.dumps(space.to_jsonable([sample])))
    )[0]
    assert data_equivalence(sample, recovered)
    flat_space = flatten_space(space)
    assert (flat_space.num_nodes, flat_space.num_edges) == (3, num_edges)
    flat_sample = flatten(space, sample)
    assert flat_sample in flat_space
    assert data_equivalence(unflatten(space, flat_sample), sample)
    assert create_zero_array(space) in space
    empty = create_empty_array(space, n=2)
    assert len(empty) == 2
    assert empty[0].nodes[0]["kind"].shape == (3,)
    assert (
        empty[0].edges is None
        if num_edges == 0
        else empty[0].edges["weight"].shape == (2,)
    )
    other = Graph(
        deepcopy(space.node_space),
        deepcopy(space.edge_space),
        num_nodes=4,
        num_edges=num_edges,
    )
    assert space != other
    assert not is_space_dtype_shape_equiv(space, other)
    assert is_space_dtype_shape_equiv(space, deepcopy(space))
    assert "num_nodes=3" in repr(space)
    assert "num_nodes" not in repr(Graph(Discrete(2), None))
