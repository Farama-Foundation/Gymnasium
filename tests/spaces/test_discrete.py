from copy import deepcopy

import numpy as np

from gymnasium.spaces import Discrete


def test_space_legacy_pickling():
    """Test the legacy pickle of Discrete that is missing the `start` parameter."""
    # Test that start is corrected passed
    space = Discrete(1, start=2)
    state = space.__dict__

    new_space = Discrete(1)
    new_space.__setstate__(state)
    assert new_space == space
    assert new_space.start == 2

    legacy_space = Discrete(1)
    legacy_state = deepcopy(legacy_space.__dict__)
    del legacy_state["start"]

    new_legacy_space = Discrete(2)
    new_legacy_space.__setstate__(legacy_state)
    assert new_legacy_space == legacy_space
    assert new_legacy_space.start == 0


def test_sample_mask():
    space = Discrete(4, start=2)
    assert 2 <= space.sample() < 6
    assert space.sample(mask=np.array([0, 1, 0, 0], dtype=np.int8)) == 3
    assert space.sample(mask=np.array([0, 0, 0, 0], dtype=np.int8)) == 2
    assert space.sample(mask=np.array([0, 1, 0, 1], dtype=np.int8)) in [3, 5]
