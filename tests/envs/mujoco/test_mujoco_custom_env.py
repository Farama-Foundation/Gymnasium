__credits__ = ["Kallinteris-Andreas"]

import os
import warnings

import numpy as np
import pytest

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.error import Error
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env


class PointEnv(MujocoEnv, utils.EzPickle):
    """
    A simple mujoco env to test third party mujoco env, using the `Gymnasium.MujocoEnv` environment API.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(self, xml_file="point.xml", frame_skip=1, **kwargs):
        utils.EzPickle.__init__(self, xml_file, frame_skip, **kwargs)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config={},
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]

        observation = self._get_obs()
        reward = x_position_after - x_position_before
        info = {}

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        return np.concatenate((position, velocity))

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {"works": True}


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]


@pytest.mark.parametrize("frame_skip", [1, 2, 3, 4, 5])
def test_frame_skip(frame_skip):
    """verify that custom envs work with different `frame_skip` values"""
    env = PointEnv(frame_skip=frame_skip)

    # Test if env adheres to Gym API
    with warnings.catch_warnings(record=True) as w:
        check_env(env.unwrapped, skip_render_check=True)
        env.close()
    for warning in w:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning.message}")


def test_xml_file():
    """Verify that the loading of a custom XML file works"""
    relative_path = "./tests/envs/mujoco/assets/walker2d_v5_uneven_feet.xml"
    env = PointEnv(xml_file=relative_path).unwrapped
    assert isinstance(env, MujocoEnv)
    assert env.data.qpos.size == 9

    full_path = os.getcwd() + "/tests/envs/mujoco/assets/walker2d_v5_uneven_feet.xml"
    env = PointEnv(xml_file=full_path).unwrapped
    assert isinstance(env, MujocoEnv)
    assert env.data.qpos.size == 9

    # note can not test user home path (with '~') because github CI does not have a home folder


def test_reset_info():
    """Verify that the environment returns info at `reset()`"""
    env = PointEnv()

    _, info = env.reset()
    assert info["works"] is True


OFFSCREEN_BUFFER_XML = """<mujoco model="offscreen_buffer_env">
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body name="torso" pos="0 0 1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def test_offscreen_framebuffer_not_shrunk(tmp_path):
    """Verify that an XML-declared offscreen framebuffer is not shrunk.

    Regression test for https://github.com/Farama-Foundation/Gymnasium/issues/1607:
    `MujocoEnv._initialize_simulation` used to unconditionally overwrite the
    `offwidth`/`offheight` requested through `<visual><global .../></visual>`
    with the (smaller) `width`/`height` render-window size, which broke a
    user-supplied high-resolution `mujoco.Renderer`.
    """
    xml_path = tmp_path / "offscreen_buffer_env.xml"
    xml_path.write_text(OFFSCREEN_BUFFER_XML)

    # The XML requests a framebuffer larger than the default render window
    # (480x480), so the XML-declared size must survive `__init__`.
    env = PointEnv(xml_file=str(xml_path)).unwrapped
    assert env.model.vis.global_.offwidth == 1920
    assert env.model.vis.global_.offheight == 1080
    env.close()

    # When the requested render window is larger than the XML declaration, the
    # larger window size wins so that on-screen rendering still fits the buffer.
    env = PointEnv(xml_file=str(xml_path), width=2560, height=1440).unwrapped
    assert env.model.vis.global_.offwidth == 2560
    assert env.model.vis.global_.offheight == 1440
    env.close()
