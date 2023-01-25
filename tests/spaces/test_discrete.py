import numpy as np
from gym.spaces import Discrete

def test_space_legacy_pickling(legacy_state: dict) -> None:
    """Test the legacy pickle of Discrete that is missing the `start` parameter."""
    space = Discrete(1)
    space.__setstate__(legacy_state)

    assert space.shape == legacy_state["shape"]
    assert space.np_random == legacy_state["np_random"]
    assert space.n == legacy_state["n"]
    assert space.dtype == legacy_state["dtype"]

    # Test that start is missing
    assert "start" not in space.__dict__

    space.__setstate__(legacy_state)
    assert space.start == legacy_state.get("start", 0)

def test_sample_mask() -> None:
    space = Discrete(4, start=2)
    assert 2 <= space.sample() < 6
    assert space.sample(mask=np.array([0, 1, 0, 0], dtype=np.bool)) == 3
    assert space.sample(mask=np.array([0, 0, 0, 0], dtype=np.bool)) == 2
    assert space.sample(mask=np.array([0, 1, 0, 1], dtype=np.bool)) in [3, 5]
