import os

import mujoco
import numpy as np
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


@pytest.mark.parametrize(
    "env_id",
    [
        "Ant",
        "HalfCheetah",
        "Hopper",
        "Humanoid",
        "HumanoidStandup",
        "InvertedDoublePendulum",
        "InvertedPendulum",
        "Pusher",
        "Reacher",
        "Swimmer",
        "Walker2d",
    ],
)
@pytest.mark.parametrize("version", ["v4", "v5"])
def test_mujoco_metadata_render_modes(env_id, version):
    if env_id == "Pusher" and version == "v4":
        pytest.skip()
    env = gymnasium.make(f"{env_id}-{version}")
    assert env.metadata["render_modes"] == [
        "human",
        "rgb_array",
        "depth_array",
        "rgbd_tuple",
    ]
    env.close()


@pytest.mark.parametrize(
    "render_mode", ["human", "rgb_array", "depth_array", "rgbd_tuple"]
)
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


@pytest.mark.parametrize(
    "render_mode", ["human", "rgb_array", "depth_array", "rgbd_tuple"]
)
def test_add_markers(model: mujoco.MjModel, data: mujoco.MjData, render_mode: str):
    """Test that the add_markers function works correctly."""
    # initialize renderer
    renderer = ExposedViewerRenderer(
        model, data, width=DEFAULT_SIZE, height=DEFAULT_SIZE, max_geom=10
    )
    # initialize viewer via render
    viewer = renderer.get_viewer(render_mode)
    viewer.add_marker(
        pos=np.array([0, 0, 0]),
        size=np.array([1, 1, 1]),
        rgba=np.array([1, 0, 0, 1]),
    )
    args = tuple() if render_mode == "human" else (render_mode,)
    viewer.render(*args)  # We need to render to trigger the marker addition in MuJoCo
    # close viewer after usage
    assert len(viewer._markers) == 0, "Markers should be cleared after rendering."
    viewer.close()


@pytest.mark.parametrize(
    "render_mode", ["human", "rgb_array", "depth_array", "rgbd_tuple"]
)
def test_camera_id(render_mode: str):
    """Assert that the camera_id parameter works correctly."""
    env_a = gymnasium.make("Ant-v5", camera_id=0, render_mode=render_mode).unwrapped
    env_b = gymnasium.make("Ant-v5", camera_id=0, render_mode=render_mode).unwrapped
    env_c = gymnasium.make("Ant-v5", camera_id=-1, render_mode=render_mode).unwrapped

    assert env_a.mujoco_renderer.camera_id == env_b.mujoco_renderer.camera_id
    assert env_a.mujoco_renderer.camera_id != env_c.mujoco_renderer.camera_id

    if render_mode == "rgbd_tuple":
        rgb_a, depth_a = env_a.render()
        rgb_b, depth_b = env_b.render()
        rgb_c, depth_c = env_c.render()
        assert (rgb_a == rgb_b).all()
        assert (depth_a == depth_b).all()
        assert (rgb_a != rgb_c).any()
        assert (depth_a != depth_c).any()

    elif render_mode != "human":
        assert (env_a.render() == env_b.render()).all()
        assert (env_a.render() != env_c.render()).any()


def test_rgbd_tuple():
    """Assert that rgbd_tuple is the proper combination of rgb and depth images as tuple"""
    env_a = gymnasium.make("Ant-v5", render_mode="rgbd_tuple").unwrapped
    env_b = gymnasium.make("Ant-v5", render_mode="rgb_array").unwrapped
    env_c = gymnasium.make("Ant-v5", render_mode="depth_array").unwrapped

    rgb_a, depth_a = env_a.render()
    rgb_b = env_b.render()
    depth_c = env_c.render()

    assert isinstance(rgb_a, np.ndarray)
    assert isinstance(depth_c, np.ndarray)
    assert rgb_a.dtype == np.uint8
    assert depth_a.dtype == np.float32
    assert rgb_a.ndim == 3
    assert depth_a.ndim == 2

    assert (rgb_a == rgb_b).all()
    assert (depth_a == depth_c).all()
