import re

import pytest

from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map


@pytest.mark.parametrize(
    "desc",
    [
        ["SF"] + ["FF"] * 14 + ["FG"],
        ["S" + "F" * 15, "F" * 15 + "G"],
    ],
)
def test_sprites_fit_rectangular_cells(desc, monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    env = FrozenLakeEnv(desc=desc, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        env.render()

        for sprite in (
            env.agent_img,
            env.goal_flag_img,
            env.letter_s_img,
            env.letter_g_img,
        ):
            assert sprite.get_width() <= env.cell_size[0]
            assert sprite.get_height() <= env.cell_size[1]
    finally:
        env.close()


@pytest.mark.parametrize(
    "size, p, message",
    [
        (1, 0.8, "`size` must be at least 2, got 1"),
        (0, 0.8, "`size` must be at least 2, got 0"),
        (8, 0.0, "`p` must be greater than 0, got 0.0"),
    ],
    ids=["size-1", "size-0", "p-0"],
)
def test_generate_random_map_rejects_impossible_parameters(size, p, message):
    # No valid map exists for these values. Before the check, size 1 and p 0 looped forever.
    with pytest.raises(ValueError, match=re.escape(message)):
        generate_random_map(size=size, p=p, seed=0)
