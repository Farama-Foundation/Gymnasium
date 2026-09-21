"""Fixed graph buffers retain live views and reject malformed worker observations."""

import multiprocessing as mp
from contextlib import closing
from copy import deepcopy
from functools import partial

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, Graph, GraphInstance, Tuple
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from gymnasium.vector.utils import (
    create_shared_memory,
    read_from_shared_memory,
    write_to_shared_memory,
)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.uint8, np.bool_])
@pytest.mark.parametrize("num_edges", [0, 2])
def test_graph_live_buffers(dtype, num_edges):
    """Previously read nested arrays see subsequent writes to only their worker."""
    space = Graph(
        Dict({"value": Box(0, 1, (2,), dtype=dtype), "id": Discrete(3)}),
        Tuple((Discrete(3), Box(0, 1, (), dtype=dtype))),
        num_nodes=3,
        num_edges=num_edges,
    )
    memory = create_shared_memory(space, 2, mp.get_context("spawn"))
    views = read_from_shared_memory(space, memory, 2)
    initial, updated = space.sample(), space.sample()
    initial.nodes["id"][:] = 0
    updated.nodes["id"][:] = 2
    for index in range(2):
        write_to_shared_memory(space, index, initial, memory)
    snapshot = deepcopy(views)
    if num_edges:
        updated = updated._replace(edge_links=updated.edge_links.astype(np.int64))
    write_to_shared_memory(space, 1, updated, memory)
    assert data_equivalence(views[0], snapshot[0])
    assert data_equivalence(views[1].nodes, updated.nodes)
    assert views[1] in space
    assert isinstance(views[1].nodes["id"], np.ndarray)
    assert views[1].nodes["value"].dtype == dtype
    assert not np.shares_memory(views[0].nodes["value"], views[1].nodes["value"])
    if num_edges:
        assert views[1].edge_links.dtype == np.int32
        np.testing.assert_array_equal(views[1].edge_links, updated.edge_links)
    else:
        assert views[1].edges is views[1].edge_links is None


def test_graph_invalid_writes_are_atomic():
    """A bad last feature or link cannot leave the first features partially changed."""
    space = Graph(
        Dict({"a": Box(0, 1, (2,)), "b": Discrete(3)}),
        Discrete(3),
        num_nodes=3,
        num_edges=2,
    )
    memory = create_shared_memory(space, 2)
    views = read_from_shared_memory(space, memory, 2)
    for index in range(2):
        write_to_shared_memory(space, index, space.sample(), memory)
    before = deepcopy(views)
    candidate = space.sample()
    invalid = [
        tuple(candidate),
        candidate._replace(nodes={"a": candidate.nodes["a"]}),
        candidate._replace(
            nodes={**candidate.nodes, "b": np.ones(3, dtype=np.float64)}
        ),
        candidate._replace(nodes={**candidate.nodes, "a": candidate.nodes["a"][:1]}),
        candidate._replace(edges=candidate.edges[:1]),
        candidate._replace(edge_links=np.full((2, 2), 2**40, dtype=np.int64)),
        candidate._replace(edge_links=np.full((2, 2), -1, dtype=np.int64)),
        candidate._replace(edge_links=np.zeros((2, 2), dtype=np.float32)),
        candidate._replace(edge_links=np.zeros((1, 2), dtype=np.int32)),
    ]
    for value in invalid:
        with pytest.raises(ValueError):
            write_to_shared_memory(space, 1, value, memory)
        assert data_equivalence(views, before)
    for index in (-1, 2):
        with pytest.raises(IndexError):
            write_to_shared_memory(space, index, candidate, memory)
        assert data_equivalence(views, before)


class FixedGraphEnv(gym.Env):
    """Deterministic changing graph, with staggered endings for vector lifecycle tests."""

    def __init__(self, length=2, nesting="graph", invalid=False):
        self.graph_space = Graph(
            Dict({"step": Discrete(100), "value": Box(0, 100, (2,), dtype=np.float16)}),
            Tuple((Discrete(100),)),
            num_nodes=3,
            num_edges=2,
        )
        self.nesting = nesting
        self.observation_space = self.graph_space
        if nesting == "dict":
            self.observation_space = Dict({"graph": self.graph_space})
        elif nesting == "tuple":
            self.observation_space = Tuple((self.graph_space,))
        self.action_space = Discrete(2)
        self.length, self.invalid = length, invalid
        self.count = 0

    def observation(self):
        """Encode the raw step count into all graph leaves."""
        graph = GraphInstance(
            {
                "step": np.full(3, self.count, dtype=np.int64),
                "value": np.full((3, 2), self.count, dtype=np.float16),
            },
            (np.full(2, self.count, dtype=np.int64),),
            np.full((2, 2), self.count % 3, dtype=np.int32),
        )
        if self.invalid and self.count:
            graph = graph._replace(edge_links=np.full((2, 2), 10, dtype=np.int64))
        if self.nesting == "dict":
            return {"graph": graph}
        if self.nesting == "tuple":
            return (graph,)
        return graph

    def reset(self, *, seed=None, options=None):
        """Reset the counter without changing the graph's layout."""
        super().reset(seed=seed)
        self.count = 0
        return self.observation(), {}

    def step(self, action):
        """Advance once and terminate at the configured count."""
        self.count += 1
        return (
            self.observation(),
            1.0,
            self.count == self.length,
            False,
            {"count": self.count},
        )


@pytest.mark.parametrize("nesting", ["graph", "dict", "tuple"])
@pytest.mark.parametrize("copy", [False, True])
def test_graph_spawn_copy_and_nesting(nesting, copy):
    """A single initialization read is enough to receive all later worker updates."""
    with closing(
        AsyncVectorEnv(
            [partial(FixedGraphEnv, length=10, nesting=nesting)] * 2,
            context="spawn",
            copy=copy,
        )
    ) as env:
        initial, _ = env.reset(seed=2)
        before = deepcopy(initial)
        updated, *_ = env.step(np.zeros(2, dtype=np.int64))
        assert updated in env.observation_space
        assert data_equivalence(initial, before) is copy
        graphs = (
            updated["graph"]
            if nesting == "dict"
            else updated[0]
            if nesting == "tuple"
            else updated
        )
        for graph in graphs:
            np.testing.assert_array_equal(graph.nodes["step"], [1, 1, 1])
            np.testing.assert_array_equal(graph.edges[0], [1, 1])
            assert graph.edge_links.dtype == np.int32


@pytest.mark.parametrize("mode", list(AutoresetMode))
def test_graph_spawn_autoreset_and_partial_reset(mode):
    """Compare shared transport to the pipe path across terminal observations and reset masks."""
    factories = [partial(FixedGraphEnv, length=2), partial(FixedGraphEnv, length=4)]
    with (
        closing(
            AsyncVectorEnv(
                factories, shared_memory=True, context="spawn", autoreset_mode=mode
            )
        ) as shared,
        closing(
            AsyncVectorEnv(
                factories, shared_memory=False, context="spawn", autoreset_mode=mode
            )
        ) as pipe,
    ):
        a, b = shared.reset(seed=4), pipe.reset(seed=4)
        assert data_equivalence(a, b)
        for _ in range(7):
            a, b = (
                shared.step(np.zeros(2, dtype=np.int64)),
                pipe.step(np.zeros(2, dtype=np.int64)),
            )
            assert data_equivalence(a, b)
            if mode == AutoresetMode.DISABLED and np.any(a[2] | a[3]):
                options = {"reset_mask": a[2] | a[3]}
                assert data_equivalence(
                    shared.reset(options=deepcopy(options)),
                    pipe.reset(options=deepcopy(options)),
                )


def test_graph_spawn_error_cleanup():
    """A worker's invalid graph propagates as an error and all processes close."""
    env = AsyncVectorEnv([partial(FixedGraphEnv, invalid=True)], context="spawn")
    processes = tuple(env.processes)
    try:
        env.reset()
        with pytest.raises(ValueError, match="edge_links"):
            env.step(np.zeros(1, dtype=np.int64))
    finally:
        env.close(terminate=True)
    assert all(not process.is_alive() for process in processes)


def test_dynamic_graph_still_rejected():
    """Only explicitly constrained graphs are eligible for static buffers."""
    with pytest.raises(TypeError, match="dynamic shape"):
        create_shared_memory(Graph(Discrete(3), None))
