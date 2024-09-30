import os

import mujoco
import pytest

import gymnasium
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer, OffScreenViewer


ASSET_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "walker2d_v5_uneven_feet.xml"
)
DEFAULT_MAX_GEOMS = 1000


class ExposedViewerRenderer(MujocoRenderer):
    """Expose the viewer for testing to avoid warnings."""

    def get_viewer(self, render_mode: str):
        return self._get_viewer(render_mode)


@pytest.fixture(scope="module")
def model():
    """Initialize a model."""
    model = mujoco.MjModel.from_xml_path(ASSET_PATH)
    model.vis.global_.offwidth = DEFAULT_SIZE
    model.vis.global_.offheight = DEFAULT_SIZE
    return model


@pytest.fixture(scope="module")
def data(model):
    """Initialize data."""
    return mujoco.MjData(model)


@pytest.mark.parametrize("width", [10, 100, 200, 480])
@pytest.mark.parametrize("height", [10, 100, 200, 480])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_offscreen_viewer_custom_dimensions(
    model: mujoco.MjModel, data: mujoco.MjData, width: int, height: int
):
    """Test that the offscreen viewer has the correct dimensions."""

    # initialize viewer
    viewer = OffScreenViewer(model, data, width=width, height=height)

    # assert viewer dimensions
    assert viewer.viewport.width == width
    assert viewer.viewport.height == height

    # check that the render method returns an image of the correct shape
    img = viewer.render(render_mode="rgb_array")
    assert img.shape == (height, width, 3)

    # close viewer after usage
    viewer.close()


@pytest.mark.parametrize("render_mode", ["human", "rgb_array", "depth_array"])
@pytest.mark.parametrize("max_geom", [10, 100, 1000, 10000])
def test_max_geom_attribute(
    model: mujoco.MjModel, data: mujoco.MjData, render_mode: str, max_geom: int
):
    """Test that the max_geom attribute is set correctly."""

    # initialize renderer
    renderer = ExposedViewerRenderer(
        model, data, width=DEFAULT_SIZE, height=DEFAULT_SIZE, max_geom=max_geom
    )

    # assert max_geom attribute
    assert renderer.max_geom == max_geom

    # initialize viewer via render
    viewer = renderer.get_viewer(render_mode)

    # assert that max_geom is set correctly in the viewer scene
    assert viewer.scn.maxgeom == max_geom

    # close viewer after usage
    viewer.close()


@pytest.mark.parametrize("render_mode", ["human", "rgb_array", "depth_array"])
def test_camera_id(render_mode: str):
    """Assert that the camera_id parameter works correctly."""
    env_a = gymnasium.make("Ant-v5", camera_id=0, render_mode=render_mode).unwrapped
    env_b = gymnasium.make("Ant-v5", camera_id=0, render_mode=render_mode).unwrapped
    env_c = gymnasium.make("Ant-v5", camera_id=-1, render_mode=render_mode).unwrapped

    assert env_a.mujoco_renderer.camera_id == env_b.mujoco_renderer.camera_id
    assert env_a.mujoco_renderer.camera_id != env_c.mujoco_renderer.camera_id

    if render_mode != "human":
        assert (env_a.render() == env_b.render()).all()
        assert (env_a.render() != env_c.render()).any()
