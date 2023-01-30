import numpy as np
import pytest

import gymnasium as gym
from gymnasium.logger import warn
from tests.envs.utils import all_testing_env_ids


def check_rendered(rendered_frame, mode: str):
    """Check that the rendered frame is as expected."""
    if mode == "rgb_array_list":
        assert isinstance(rendered_frame, list)
        for frame in rendered_frame:
            check_rendered(frame, "rgb_array")
    elif mode == "rgb_array":
        assert isinstance(rendered_frame, np.ndarray)
        assert len(rendered_frame.shape) == 3
        assert rendered_frame.shape[2] == 3
        assert np.all(rendered_frame >= 0) and np.all(rendered_frame <= 255)
    elif mode == "ansi":
        assert isinstance(rendered_frame, str)
        assert len(rendered_frame) > 0
    elif mode == "state_pixels_list":
        assert isinstance(rendered_frame, list)
        for frame in rendered_frame:
            check_rendered(frame, "rgb_array")
    elif mode == "state_pixels":
        check_rendered(rendered_frame, "rgb_array")
    elif mode == "depth_array_list":
        assert isinstance(rendered_frame, list)
        for frame in rendered_frame:
            check_rendered(frame, "depth_array")
    elif mode == "depth_array":
        assert isinstance(rendered_frame, np.ndarray)
        assert len(rendered_frame.shape) == 2
    else:
        warn(
            f"Unknown render mode: {mode}, cannot check that the rendered data is correct. Add case to `check_rendered`"
        )


# We do not check render_mode for some mujoco envs and any old Gym environment wrapped by `GymEnvironment`
render_mode_env_ids = [
    spec
    for spec in all_testing_env_ids
    if "mujoco" not in spec.entry_point or "v4" in spec.id
]


@pytest.mark.parametrize("env_id", render_mode_env_ids, ids=render_mode_env_ids)
def test_render_modes(env_id):
    """There is a known issue where rendering a mujoco environment then mujoco-py will cause an error on non-mac based systems.

    Therefore, we are only testing with mujoco environments.
    """
    env = gym.make(env_id)

    assert "rgb_array" in env.metadata["render_modes"]

    for mode in env.metadata["render_modes"]:
        if mode != "human":
            new_env = gym.make(env_id, render_mode=mode)

            new_env.reset()
            rendered = new_env.render()
            check_rendered(rendered, mode)

            new_env.step(new_env.action_space.sample())
            rendered = new_env.render()
            check_rendered(rendered, mode)

            new_env.close()
    env.close()
