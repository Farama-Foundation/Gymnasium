# Composite Spaces

Graph observations may have variable counts by default. Supply both `num_nodes`
and `num_edges` to declare fixed counts while allowing features and connectivity
to change:

```python
from gymnasium.spaces import Box, Graph

space = Graph(Box(-1, 1, (3,)), Box(0, 1, (2,)), num_nodes=32, num_edges=64)
observation = space.sample()
assert observation in space
```

Fixed-count graphs support the standard `Box`, `Discrete`, `MultiBinary` and
`MultiDiscrete` spaces, including nonempty nested `Dict`/`Tuple` compositions.
Custom subclasses and other feature spaces are not supported in fixed mode.
Counts are exact constraints, not maximum sizes. The node count must be a
positive integer that fits int32, and the edge count a nonnegative integer;
booleans are not accepted. The two counts are independent: self-loops and
repeated links are allowed, and connectivity may change between observations.

Zero edges are represented by `edges=None` and `edge_links=None`;
when `edge_space=None`, use `num_edges=0`. Supplying conflicting counts to
`sample()` is an error. Counts are preserved by JSON roundtrips, structured
flattening and batching. Flattening still returns a graph, not a single array;
vector observations remain a tuple of graphs. Fixed-count graphs can be used
with `SyncVectorEnv` or `AsyncVectorEnv(shared_memory=False)`.

For fixed graphs, sampling masks follow the batched feature spaces. For example,
with `Discrete` features, provide a tuple of masks with one entry per node or
edge. This also applies when `sample()` uses the space's declared counts:

```python
import numpy as np
from gymnasium.spaces import Discrete

space = Graph(Discrete(3), Discrete(2), num_nodes=3, num_edges=2)
node_masks = tuple(np.eye(3, dtype=np.int8))
edge_masks = tuple(np.eye(2, dtype=np.int8))
observation = space.sample(mask=(node_masks, edge_masks))
assert observation.nodes.tolist() == [0, 1, 2]
assert observation.edges.tolist() == [0, 1]
```

Probability masks use the same layout. For zero-edge graphs, the edge mask can
be `None` or an empty tuple. Dynamic graphs retain their existing sampling
behavior: the default node count is 10, and omitting `num_edges` broadcasts a
single edge mask over the randomly sampled edges. Previously pickled dynamic
graphs load without count constraints.


```{eval-rst}
.. autoclass:: gymnasium.spaces.Dict

    .. automethod:: gymnasium.spaces.Dict.sample
    .. automethod:: gymnasium.spaces.Dict.seed

.. autoclass:: gymnasium.spaces.Tuple

    .. automethod:: gymnasium.spaces.Tuple.sample
    .. automethod:: gymnasium.spaces.Tuple.seed

.. autoclass:: gymnasium.spaces.Sequence

    .. automethod:: gymnasium.spaces.Sequence.sample
    .. automethod:: gymnasium.spaces.Sequence.seed

.. autoclass:: gymnasium.spaces.Graph

    .. automethod:: gymnasium.spaces.Graph.sample
    .. automethod:: gymnasium.spaces.Graph.seed

.. autoclass:: gymnasium.spaces.OneOf

    .. automethod:: gymnasium.spaces.OneOf.sample
    .. automethod:: gymnasium.spaces.OneOf.seed
```
