import pytest

from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv


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
