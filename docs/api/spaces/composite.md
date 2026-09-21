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

Fixed-count graphs support numerical spaces and nonempty nested Dict/Tuple
features. Zero edges are represented by `edges=None` and `edge_links=None`;
when `edge_space=None`, use `num_edges=0`. Supplying conflicting counts to
`sample()` is an error. Counts are preserved by JSON roundtrips, structured
flattening and batching. Flattening still returns a graph, not a single array;
vector observations remain a tuple of graphs. Fixed-count graphs can be used
with `SyncVectorEnv` or `AsyncVectorEnv(shared_memory=False)`.


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
