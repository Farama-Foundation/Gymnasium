"""Implementation of a space that represents graph information where nodes and edges can be represented with euclidean space."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.spaces.space import Space


class GraphInstance(NamedTuple):
    """A Graph space instance.

    * nodes (Iterable): an (n x ...) sized array representing the features for n nodes, (...) must adhere to the shape of the node space.
    * edges (Optional[Iterable]): an (m x ...) sized array representing the features for m edges, (...) must adhere to the shape of the edge space.
    * edge_links (Optional[np.ndarray]): an (m x 2) sized array of ints representing the indices of the two nodes that each edge connects.
    """

    nodes: Iterable[Any]
    edges: Iterable[Any] | None
    edge_links: NDArray[Any] | None


class Graph(Space[GraphInstance]):
    r"""A space representing graph information as a series of ``nodes`` connected with ``edges`` according to an adjacency matrix represented as a series of ``edge_links``.

    By default, observations may contain different numbers of nodes and edges.
    Set both ``num_nodes`` and ``num_edges`` to constrain their counts while
    allowing feature values and connectivity to change. Fixed-count graphs
    support Box, Discrete, MultiBinary and MultiDiscrete feature spaces and
    nonempty Dict/Tuple compositions of these spaces.
    Zero-edge graphs use ``edges=None`` and ``edge_links=None``.

    Example:
        >>> from gymnasium.spaces import Graph, Box, Discrete
        >>> observation_space = Graph(node_space=Box(low=-100, high=100, shape=(3,)), edge_space=Discrete(3), seed=123)
        >>> observation_space.sample(num_nodes=4, num_edges=8)
        GraphInstance(nodes=array([[-50.143734, -89.37025 , -42.120003],
               [ 31.519672, -96.5817  ,  85.16244 ],
               [ 97.65833 ,  92.20077 ,  75.31062 ],
               [-21.80443 , -73.790115,  39.97857 ]], dtype=float32), edges=array([1, 0, 2, 0, 2, 1, 1, 2]), edge_links=array([[0, 2],
               [2, 0],
               [3, 0],
               [1, 0],
               [1, 0],
               [1, 3],
               [1, 3],
               [1, 1]], dtype=int32))
    """

    node_space: Space[Any]
    edge_space: Space[Any] | None
    num_nodes: int | None
    num_edges: int | None

    def __init__(
        self,
        node_space: Space[Any],
        edge_space: None | Space[Any],
        seed: int | np.random.Generator | None = None,
        *,
        num_nodes: int | None = None,
        num_edges: int | None = None,
    ) -> None:
        r"""Constructor of :class:`Graph`.

        The argument ``node_space`` specifies the base space that each node feature will use.

        The argument ``edge_space`` specifies the base space that each edge feature will use.

        Args:
            node_space (Space[Any]): space of the node features.
            edge_space (None | Space[Any]): space of the edge features.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
            num_nodes: A fixed positive integer node count, limited to int32,
                or None for a dynamic graph.
            num_edges: A fixed nonnegative integer edge count. Must be supplied together
                with num_nodes, and must be zero when edge_space is None.

        Raises:
            TypeError: A fixed count is not an integer (booleans are not accepted),
                or a fixed feature space is unsupported. Fixed features must use
                the standard numerical spaces above, or nonempty Dict/Tuple
                compositions of them; custom subclasses are not supported.
            ValueError: Only one count is supplied, a count is out of range,
                or edges are requested without an edge space.
        """
        if (num_nodes is None) != (num_edges is None):
            raise ValueError("num_nodes and num_edges must be provided together.")
        if num_nodes is not None:
            assert num_edges is not None
            self._validate_count(num_nodes, "num_nodes")
            self._validate_count(num_edges, "num_edges")
            num_nodes, num_edges = int(num_nodes), int(num_edges)
            if edge_space is None and num_edges != 0:
                raise ValueError("num_edges must be zero when edge_space is None.")
            self._validate_fixed_feature_space(node_space, "node_space")
            if edge_space is not None:
                self._validate_fixed_feature_space(edge_space, "edge_space")
        self.num_nodes, self.num_edges = num_nodes, num_edges
        self.node_space = node_space
        self.edge_space = edge_space

        self.batch_node_space = gym.vector.utils.batch_space(
            node_space, n=num_nodes or 1
        )
        if edge_space is not None:
            self.batch_edge_space = gym.vector.utils.batch_space(
                edge_space, n=num_edges or 1
            )
        else:
            self.batch_edge_space = None

        super().__init__(None, None, seed)

    @staticmethod
    def _validate_count(value: Any, name: str) -> None:
        """Validate a fixed graph count before sampling or advancing any RNG."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer, got {value!r}.")
        minimum = 1 if name == "num_nodes" else 0
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {value}.")
        if name == "num_nodes" and value > np.iinfo(np.int32).max:
            raise ValueError("num_nodes must fit the int32 edge-link representation.")

    @staticmethod
    def _validate_fixed_feature_space(space: Space[Any], path: str) -> None:
        """Check that every feature leaf has a supported fixed numerical layout."""
        if type(space) in (
            gym.spaces.Box,
            gym.spaces.Discrete,
            gym.spaces.MultiBinary,
            gym.spaces.MultiDiscrete,
        ):
            return
        if type(space) is gym.spaces.Dict and space.spaces:
            for key, subspace in space.spaces.items():
                Graph._validate_fixed_feature_space(subspace, f"{path}[{key!r}]")
            return
        if type(space) is gym.spaces.Tuple and space.spaces:
            for index, subspace in enumerate(space.spaces):
                Graph._validate_fixed_feature_space(subspace, f"{path}[{index}]")
            return
        raise TypeError(
            f"Fixed-count Graph features require numerical spaces or nonempty "
            f"Dict/Tuple compositions; unsupported {path}: {space!r}."
        )

    @property
    def is_np_flattenable(self) -> Literal[False]:
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return False

    def seed(
        self, seed: int | tuple[int, int] | tuple[int, int, int] | None = None
    ) -> tuple[int, int] | tuple[int, int, int]:
        """Seeds the PRNG of this space and node / edge subspace.

        Depending on the type of seed, the subspaces will be seeded differently

        * ``None`` - The root, node and edge spaces PRNG are randomly initialized
        * ``Int`` - The integer is used to seed the :class:`Graph` space that is used to generate seed values for the node and edge subspaces.
        * ``Tuple[int, int]`` - Seeds the :class:`Graph` and node subspace with a particular value. Only if edge subspace isn't specified
        * ``Tuple[int, int, int]`` - Seeds the :class:`Graph`, node and edge subspaces with a particular value.

        Args:
            seed: An optional int or tuple of ints for this space and the node / edge subspaces. See above for more details.

        Returns:
            A tuple of two or three ints depending on if the edge subspace is specified.
        """
        if seed is None:
            if self.edge_space is None:
                return super().seed(None), self.node_space.seed(None)
            else:
                return (
                    super().seed(None),
                    self.node_space.seed(None),
                    self.edge_space.seed(None),
                )
        elif isinstance(seed, int):
            if self.edge_space is None:
                super_seed = super().seed(seed)
                node_seed = int(self.np_random.integers(np.iinfo(np.int32).max))
                # this is necessary such that after int or list/tuple seeding, the Graph PRNG are equivalent
                super().seed(seed)
                return super_seed, self.node_space.seed(node_seed)
            else:
                super_seed = super().seed(seed)
                node_seed, edge_seed = self.np_random.integers(
                    np.iinfo(np.int32).max, size=(2,)
                )
                # this is necessary such that after int or list/tuple seeding, the Graph PRNG are equivalent
                super().seed(seed)
                return (
                    super_seed,
                    self.node_space.seed(int(node_seed)),
                    self.edge_space.seed(int(edge_seed)),
                )
        elif isinstance(seed, (list, tuple)):
            if self.edge_space is None:
                if len(seed) != 2:
                    raise ValueError(
                        f"Expects a tuple of two values for Graph and node space, actual length: {len(seed)}"
                    )

                return super().seed(seed[0]), self.node_space.seed(seed[1])
            else:
                if len(seed) != 3:
                    raise ValueError(
                        f"Expects a tuple of three values for Graph, node and edge space, actual length: {len(seed)}"
                    )

                return (
                    super().seed(seed[0]),
                    self.node_space.seed(seed[1]),
                    self.edge_space.seed(seed[2]),  # ty:ignore[index-out-of-bounds]
                )
        else:
            raise TypeError(
                f"Expects `None`, int or tuple of ints, actual type: {type(seed)}"
            )

    def sample(
        self,
        mask: (
            tuple[
                NDArray[Any] | tuple[Any, ...] | None,
                NDArray[Any] | tuple[Any, ...] | None,
            ]
        )
        | None = None,
        probability: (
            tuple[
                NDArray[Any] | tuple[Any, ...] | None,
                NDArray[Any] | tuple[Any, ...] | None,
            ]
        )
        | None = None,
        num_nodes: int | None = None,
        num_edges: int | None = None,
    ) -> GraphInstance:
        """Sample a graph using its fixed counts or the requested dynamic counts.

        For fixed-count graphs, masks follow the batched node and edge spaces.
        For example, Discrete features require one mask per node and one per
        edge, even when counts are omitted from this call. Zero-edge graphs
        accept None or an empty tuple as the edge mask.

        For dynamic graphs, omitting ``num_edges`` samples an edge count and
        repeats a single edge mask for each sampled edge. Supplying a count
        requires a mask for the batched edge space instead. These rules apply
        to both ``mask`` and ``probability``.

        Args:
            mask: An optional tuple of optional node and edge mask
                (Box spaces don't support sample masks).
            probability: An optional tuple of optional node and edge probability mask
                (Box spaces don't support sample probability masks).
            num_nodes: The number of nodes to sample. Defaults to the fixed count,
                or 10 for a dynamic graph. An explicit count must match a fixed count.
            num_edges: The number of edges to sample. Defaults to the fixed count,
                or the existing random edge-count rule for a dynamic graph.

        Returns:
            A :class:`GraphInstance` with attributes `.nodes`, `.edges`, and `.edge_links`.
        """
        if self.num_nodes is not None:
            for name, requested, expected in (
                ("num_nodes", num_nodes, self.num_nodes),
                ("num_edges", num_edges, self.num_edges),
            ):
                if requested is not None:
                    self._validate_count(requested, name)
                    if requested != expected:
                        raise ValueError(
                            f"{name} must match the fixed count {expected}, got {requested}."
                        )
            num_nodes, num_edges = self.num_nodes, self.num_edges
        elif num_nodes is None:
            num_nodes = 10
        assert num_nodes > 0, (
            f"The number of nodes is expected to be greater than 0, actual value: {num_nodes}"
        )

        if mask is not None and probability is not None:
            raise ValueError(
                f"Only one of `mask` or `probability` can be provided, actual values: mask={mask}, probability={probability}"
            )
        elif mask is not None:
            node_space_mask, edge_space_mask = mask
            mask_type = "mask"
        elif probability is not None:
            node_space_mask, edge_space_mask = probability
            mask_type = "probability"
        else:
            node_space_mask = edge_space_mask = mask_type = None

        if self.num_edges == 0 and edge_space_mask is not None:
            if not isinstance(edge_space_mask, tuple) or len(edge_space_mask) != 0:
                raise ValueError(
                    "A fixed zero-edge graph requires an empty or None edge mask."
                )

        # we only have edges when we have at least 2 nodes
        if num_edges is None:
            if num_nodes > 1:
                # maximal number of edges is `n*(n-1)` allowing self connections and two-way is allowed
                num_edges = int(self.np_random.integers(num_nodes * (num_nodes - 1)))
            else:
                num_edges = 0

            if edge_space_mask is not None:
                edge_space_mask = tuple(edge_space_mask for _ in range(num_edges))
        else:
            if self.edge_space is None and self.num_nodes is None:
                gym.logger.warn(
                    f"The number of edges is set ({num_edges}) but the edge space is None."
                )
            assert num_edges >= 0, (
                f"Expects the number of edges to be greater than 0, actual value: {num_edges}"
            )
        assert num_edges is not None

        if mask_type is not None:
            node_sample_kwargs = {mask_type: node_space_mask}
            edge_sample_kwargs = {mask_type: edge_space_mask}
        else:
            node_sample_kwargs = edge_sample_kwargs = {}

        # Reconstruct the batch space to preserve the existing feature-seeding behavior.
        sample_batch_node_space = gym.vector.utils.batch_space(
            self.node_space, num_nodes
        )
        sampled_nodes = sample_batch_node_space.sample(**node_sample_kwargs)
        # The batch_space function deepcopies the node_space's np_random therefore to avoid generating the same samples each time
        #   we need to get the updated np_random
        self.node_space.np_random.random()

        # It is valid to sample one node and one edge (self loop)
        if num_nodes >= 1 and num_edges >= 1 and self.edge_space is not None:
            sample_batch_edge_space = gym.vector.utils.batch_space(
                self.edge_space, num_edges
            )

            sampled_edges = sample_batch_edge_space.sample(**edge_sample_kwargs)
            self.edge_space.np_random.random()
        else:
            sampled_edges = None

        sampled_edge_links = None
        if sampled_edges is not None and num_edges > 0:
            sampled_edge_links = self.np_random.integers(
                low=0, high=num_nodes, size=(num_edges, 2), dtype=np.int32
            )

        return GraphInstance(sampled_nodes, sampled_edges, sampled_edge_links)

    def contains(self, x: GraphInstance) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if self.num_nodes is not None:
            if not isinstance(x, GraphInstance) or x.nodes not in self.batch_node_space:
                return False
            if self.num_edges == 0:
                return x.edges is None and x.edge_links is None
            return bool(
                self.batch_edge_space is not None
                and x.edges in self.batch_edge_space
                and isinstance(x.edge_links, np.ndarray)
                and x.edge_links.shape == (self.num_edges, 2)
                and np.issubdtype(x.edge_links.dtype, np.integer)
                and np.all(x.edge_links >= 0)
                and np.all(x.edge_links < self.num_nodes)
            )
        if isinstance(x, GraphInstance) and x.nodes is not None:
            # Checks the nodes
            nodes = list(gym.vector.utils.iterate(self.batch_node_space, x.nodes))
            if all(node in self.node_space for node in nodes):
                # Check the edges and edge links which are optional
                if x.edges is not None and x.edge_links is not None:
                    if self.edge_space is not None and isinstance(
                        x.edge_links, np.ndarray
                    ):
                        # Use iterate to handle all space types (Dict, Tuple, etc.)
                        edges = list(
                            gym.vector.utils.iterate(self.batch_edge_space, x.edges)
                        )
                        if all(edge in self.edge_space for edge in edges):
                            if np.issubdtype(x.edge_links.dtype, np.integer):
                                if x.edge_links.shape == (len(edges), 2):
                                    if np.all(
                                        np.logical_and(
                                            x.edge_links >= 0, x.edge_links < len(nodes)
                                        )
                                    ):
                                        return True
                else:
                    return x.edges is None and x.edge_links is None
        return False

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include ``node_space`` and ``edge_space``

        Returns:
            A representation of the space
        """
        counts = (
            f", num_nodes={self.num_nodes}, num_edges={self.num_edges}"
            if self.num_nodes is not None
            else ""
        )
        return f"Graph({self.node_space}, {self.edge_space}{counts})"

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return (
            isinstance(other, Graph)
            and self.num_nodes == other.num_nodes
            and self.num_edges == other.num_edges
            and (self.node_space == other.node_space)
            and (self.edge_space == other.edge_space)
        )

    def __setstate__(
        self, state: Iterable[tuple[str, Any]] | Mapping[str, Any]
    ) -> None:
        """Load graphs saved before count constraints were added as dynamic graphs."""
        state = dict(state)
        state.setdefault("num_nodes", None)
        state.setdefault("num_edges", None)
        super().__setstate__(state)

    def to_jsonable(
        self, sample_n: Iterable[GraphInstance]
    ) -> list[dict[str, list[float]]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        ret_n = []
        for sample in sample_n:
            ret = {"nodes": self.batch_node_space.to_jsonable([sample.nodes])}
            if sample.edges is not None and sample.edge_links is not None:
                ret["edges"] = self.batch_edge_space.to_jsonable([sample.edges])
                ret["edge_links"] = sample.edge_links.tolist()
            ret_n.append(ret)
        return ret_n

    def from_jsonable(
        self, sample_n: Sequence[dict[str, list[list[int] | list[float]]]]
    ) -> list[GraphInstance]:
        """Convert a JSONable data type to a batch of samples from this space."""
        ret: list[GraphInstance] = []
        for sample in sample_n:
            if "edges" in sample:
                assert self.edge_space is not None
                ret_n = GraphInstance(
                    self.batch_node_space.from_jsonable(sample["nodes"])[0],
                    self.batch_edge_space.from_jsonable(sample["edges"])[0],
                    np.asarray(sample["edge_links"], dtype=np.int32),
                )
            else:
                ret_n = GraphInstance(
                    self.batch_node_space.from_jsonable(sample["nodes"])[0],
                    None,
                    None,
                )
            ret.append(ret_n)
        return ret
