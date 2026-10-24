"""Tests for preserving captured frames in RenderCollection."""

import numpy as np
import pytest

from gymnasium.wrappers import RenderCollection
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("pop_frames", [False, True])
@pytest.mark.parametrize("reset_clean", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_render_collection_snapshots_reused_buffer(pop_frames, reset_clean, strided):
    """Rendering later states must not overwrite collected or returned frames."""
    buffer = np.zeros((4, 6, 3), dtype=np.uint8)
    if strided:
        buffer = buffer[:, ::2]
    env = RenderCollection(
        GenericTestEnv(
            render_func=lambda self: buffer,
            render_mode="rgb_array",
            metadata={"render_modes": ["rgb_array"]},
        ),
        pop_frames=pop_frames,
        reset_clean=reset_clean,
    )
    try:
        env.reset(seed=42)
        for value in (1, 2):
            buffer.fill(value)
            env.step(np.zeros(1, dtype=np.float32))
        # Freeze the list itself: pop_frames=False intentionally retains it.
        frames = list(env.render())
        np.testing.assert_array_equal(
            frames, [np.full_like(buffer, value) for value in (0, 1, 2)]
        )

        buffer.fill(3)
        env.step(np.zeros(1, dtype=np.float32))
        np.testing.assert_array_equal(
            frames, [np.full_like(buffer, value) for value in (0, 1, 2)]
        )
        pending_values = [3] if pop_frames else [0, 1, 2, 3]

        buffer.fill(4)
        env.reset(seed=43)
        expected_values = [4] if reset_clean else pending_values + [4]
        np.testing.assert_array_equal(
            env.render(), [np.full_like(buffer, value) for value in expected_values]
        )
    finally:
        env.close()
