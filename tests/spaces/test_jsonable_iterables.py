"""Regression tests for one-shot batches in composite-space serialization."""

import json
from copy import deepcopy

import numpy as np
import pytest

from gymnasium.spaces import Box, Dict, Discrete, Tuple
from gymnasium.utils.env_checker import data_equivalence


@pytest.mark.parametrize("composite", [Dict, Tuple])
def test_jsonable_iterator_keeps_all_fields(composite):
    """Every column retains every row when the batch is an iterator."""
    fields = [Discrete(3), Box(0, 10, shape=(1,), dtype=np.float32)]
    rows = [
        (np.int64(1), np.array([2], dtype=np.float32)),
        (np.int64(2), np.array([3], dtype=np.float32)),
    ]
    if composite is Dict:
        space = Dict(dict(zip(("first", "second"), fields, strict=True)))
        samples = [dict(zip(("first", "second"), row, strict=True)) for row in rows]
        expected = {"first": [1, 2], "second": [[2.0], [3.0]]}
    else:
        space = Tuple(fields)
        samples = rows
        expected = [[1, 2], [[2.0], [3.0]]]

    encoded = space.to_jsonable(iter(samples))
    assert encoded == expected
    decoded = space.from_jsonable(json.loads(json.dumps(encoded)))
    assert data_equivalence(samples, decoded, exact=True)


@pytest.mark.parametrize(
    "batch_type", [list, tuple, iter, lambda x: (row for row in x)]
)
@pytest.mark.parametrize("batch_size", [0, 3])
@pytest.mark.parametrize(
    "space",
    [
        Dict({"z": Discrete(4), "a": Box(-1, 1, shape=(2,))}, sort_keys=False),
        Tuple((Discrete(4), Box(-1, 1, shape=(2,)))),
        Dict(
            {
                "nested": Tuple((Discrete(4), Box(-1, 1, shape=(2,)))),
                "tail": Discrete(2),
            }
        ),
        Tuple(
            (Dict({"value": Box(-1, 1, shape=(2,)), "label": Discrete(4)}), Discrete(2))
        ),
    ],
)
def test_composite_jsonable_iterable_roundtrip(space, batch_type, batch_size):
    """Lists, tuples and one-shot batches round-trip nested samples identically."""
    space.seed(42)
    samples = [space.sample() for _ in range(batch_size)]
    original = deepcopy(samples)
    expected = space.to_jsonable(samples)

    encoded = space.to_jsonable(batch_type(samples))
    assert encoded == expected
    decoded = space.from_jsonable(json.loads(json.dumps(encoded)))

    assert len(decoded) == batch_size
    assert all(sample in space for sample in decoded)
    assert data_equivalence(original, decoded, exact=True)
    assert data_equivalence(original, samples, exact=True)
