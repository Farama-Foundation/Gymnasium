import os

import mujoco
import pytest

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer, OffScreenViewer


ASSET_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "walker2d_v5_uneven_feet.xml"
)
DEFAULT_FRAMEBUFFER_WIDTH = 480
DEFAULT_FRAMEBUFFER_HEIGHT = 480
DEFAULT_MAX_GEOMS = 1000


class ExposedViewerRenderer(MujocoRenderer):
    """Expose the viewer for testing to avoid warnings."""

    def get_viewer(self, render_mode: str):
        return self._get_viewer(render_mode)


@pytest.fixture(scope="module")
def model():
    """Initialize a model."""
    model = mujoco.MjModel.from_xml_path(ASSET_PATH)
    model.vis.global_.offwidth = DEFAULT_FRAMEBUFFER_WIDTH
    model.vis.global_.offheight = DEFAULT_FRAMEBUFFER_HEIGHT
    return model


@pytest.fixture(scope="module")
def data(model):
    """Initialize data."""
    return mujoco.MjData(model)


@pytest.mark.parametrize("width", [10, 100, 1000, None])
@pytest.mark.parametrize("height", [10, 100, 1000, None])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_offscreen_viewer_custom_dimensions(
    model: mujoco.MjModel, data: mujoco.MjData, width: int, height: int
):
    """Test that the offscreen viewer has the correct dimensions."""

    # set default buffer dimensions if no dims are given
    check_width = width or DEFAULT_FRAMEBUFFER_WIDTH
    check_height = height or DEFAULT_FRAMEBUFFER_HEIGHT

    # check for "dimensions too big" error
    if (
        check_width > DEFAULT_FRAMEBUFFER_WIDTH
        or check_height > DEFAULT_FRAMEBUFFER_HEIGHT
    ):
        # after ValueError, AttributeError is raised on call to __del__
        with pytest.raises((ValueError, AttributeError)):
            viewer = OffScreenViewer(model, data, width=width, height=height)
        return

    # initialize viewer
    viewer = OffScreenViewer(model, data, width=width, height=height)

    # assert viewer dimensions
    assert viewer.viewport.width == check_width
    assert viewer.viewport.height == check_height


@pytest.mark.parametrize("render_mode", ["human", "rgb_array", "depth_array"])
@pytest.mark.parametrize("max_geom", [10, 100, 1000, 10000])
def test_max_geom_attribute(
    model: mujoco.MjModel, data: mujoco.MjData, render_mode: str, max_geom: int
):
    """Test that the max_geom attribute is set correctly."""

    # initialize renderer
    renderer = ExposedViewerRenderer(model, data, max_geom=max_geom)

    # assert max_geom attribute
    assert renderer.max_geom == max_geom

    # initialize viewer via render
    viewer = renderer.get_viewer(render_mode)

    # assert that max_geom is set correctly in the viewer scene
    assert viewer.scn.maxgeom == max_geom
