import numpy as np
import pytest

from gymnasium.logger import warn
from tests.envs.utils import all_testing_env_specs


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
    elif mode == "rgbd_tuple":
        rendered_rgb, rendered_depth = rendered_frame
        assert isinstance(rendered_rgb, np.ndarray)
        assert isinstance(rendered_depth, np.ndarray)
    else:
        warn(
            f"Unknown render mode: {mode}, cannot check that the rendered data is correct. Add case to `check_rendered`"
        )


# We do not check render_mode for some mujoco envs and any old Gym environment wrapped by `GymEnvironment`
render_mode_env_specs = [spec for spec in all_testing_env_specs]


@pytest.mark.parametrize(
    "spec", render_mode_env_specs, ids=[spec.id for spec in render_mode_env_specs]
)
def test_render_modes(spec):
    env = spec.make()

    assert "rgb_array" in env.metadata["render_modes"]

    for mode in env.metadata["render_modes"]:
        if mode != "human":
            new_env = spec.make(render_mode=mode)

            try:
                new_env.reset()
                rendered = new_env.render()
                check_rendered(rendered, mode)

                new_env.step(new_env.action_space.sample())
                rendered = new_env.render()
                check_rendered(rendered, mode)
            except Exception as e:
                if "gladLoadGL error" in str(e):
                    pytest.skip("OpenGL not available")
                else:
                    raise
            finally:
                new_env.close()
    env.close()
