"""Human rendering routines used in both regular and vector wrappers. Should not be used directly.

* ``HumanRenderingBase`` - Provides actual rendering for wrapper.HumanRendering and wrapper.vector.HumanRendering
"""

__all__ = [
    "HumanRenderingBase",
    "T_env",
    "T_wrapper",
]

import itertools
from copy import deepcopy
from math import ceil, sqrt
from typing import Generic, TypeVar

import numpy as np

import gymnasium as gym
from gymnasium import logger
from gymnasium.core import RenderFrame
from gymnasium.error import DependencyNotInstalled


ALL_ACCEPTABLE_RENDER_MODES = ["rgb_array", "rgb_array_list"]

T_env = TypeVar("T_env", gym.Env, gym.vector.VectorEnv, covariant=True)
T_wrapper = TypeVar("T_wrapper", gym.Wrapper, gym.vector.VectorWrapper, covariant=True)


class HumanRenderingBase(Generic[T_wrapper, T_env]):
    """Abstract base class for rendering."""

    def __init__(self: T_wrapper, env: T_env, num_envs: int = 1):
        """Initialize a :class:`HumanRenderingBase` instance."""
        self.screen_size = None
        self._sub_frame_size = None
        self._scale = ceil(sqrt(num_envs))
        self.window = None
        self.clock = None

        metadata_ = deepcopy(env.metadata)
        if "human" in metadata_["render_modes"]:
            logger.warn(
                "Environment %s natively supports 'human' rendering, do not use rendering wrapper.",
                env,
            )
        else:
            metadata_["render_modes"].append("human")
        self.metadata = metadata_  # should not access env.metadata from now on

        if "render_fps" not in self.metadata:
            logger.warn(
                "The metadata 'render_fps' is required to be used with the HumanRendering wrapper"
            )

        if env.render_mode not in ALL_ACCEPTABLE_RENDER_MODES:
            logger.warn(
                f"Expected {env} 'render_mode' to be one of 'rgb_array' or 'rgb_array_list' but got '{env.render_mode}'"
            )

    def _get_render_mode(self: T_wrapper, env: T_env):
        """If HumanRendering can make use of the wrapped environments render mode, then return "human"."""
        if env.render_mode not in ALL_ACCEPTABLE_RENDER_MODES:
            return env.render_mode
        return "human"

    def _set_render_mode(self: T_wrapper, env: T_env, mode: str):
        """Sets the render mode of the wrapped environment, translate if necessary."""
        if mode == "human":
            available_modes = self._available_acceptable_render_modes()
            assert available_modes, f"{env} has no valid render modes available"
            logger.warn(
                "Setting render mode %s directly to unwrapped %s",
                available_modes[0],
                env.unwrapped,
            )
            env.unwrapped.render_mode = available_modes[0]
            return
        env.unwrapped.render_mode = mode

    def _check_config(self: T_wrapper, env: T_env):
        """Used to check the config in reset, before _render_frame."""
        if env.render_mode is None:
            available_modes = self._available_acceptable_render_modes()
            if available_modes:
                raise AssertionError(
                    "Render mode was not set for {}, set to acceptable '{}'".format(
                        env, available_modes[0]
                    )
                )
            raise AssertionError(
                "Render mode was not set for {}, there are no acceptable render modes".format(
                    env
                )
            )
        assert (
            "render_fps" in self.metadata
        ), "The metadata 'render_fps' is required to be used with the HumanRendering wrapper"
        assert (
            env.render_mode in ALL_ACCEPTABLE_RENDER_MODES
        ), f"Expected {env} 'render_mode' to be one of 'rgb_array' or 'rgb_array_list' but got '{env.render_mode}'"

    def _available_acceptable_render_modes(self):
        available_modes = [
            mode
            for mode in ALL_ACCEPTABLE_RENDER_MODES
            if mode in self.metadata.get("render_modes", [])
        ]
        return available_modes

    def _render_frame(self: T_wrapper):
        """Fetch the last frame from the base environment and render it to the screen."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[box2d]`"
            )

        def _render_sub_frame(rgb_array: RenderFrame, subframe=(0, 0)):
            """Render the subframe.

            :param rgb_array: image
            :param subframe: (row, col) tuple
            """
            assert isinstance(
                rgb_array, np.ndarray
            ), "must be a np.ndarray to be rendered"

            rgb_array = np.transpose(rgb_array, axes=(1, 0, 2))
            if self.screen_size is None:
                self.screen_size = rgb_array.shape[:2]
                self._sub_frame_size = (
                    self.screen_size[0] // self._scale,
                    self.screen_size[1] // self._scale,
                )
            assert (
                self.screen_size == rgb_array.shape[:2]
            ), f"The shape of the rgb array has changed from {self.screen_size} to {rgb_array.shape[:2]}"

            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(self.screen_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(rgb_array)
            if self._scale > 1:
                surf = pygame.transform.scale(surf, self._sub_frame_size)
            self.window.blit(
                surf,
                (
                    self._sub_frame_size[0] * subframe[1],
                    self._sub_frame_size[1] * subframe[0],
                ),
            )

        if self.env.render_mode == "rgb_array_list":
            last_rgb_array = self.env.render()
            assert isinstance(
                last_rgb_array, list
            ), "Expected render to return a list of (list of environment) RGB arrays"
            last_rgb_array = last_rgb_array[-1]
        elif self.env.render_mode == "rgb_array":
            last_rgb_array = self.env.render()
        else:
            raise Exception(
                f"Wrapped environment must have mode 'rgb_array' or 'rgb_array_list'"
                f", actual render mode: {self.env.render_mode}"
            )
        if isinstance(last_rgb_array, list):
            # A list of frames, from environments, length match self.num_envs (should now exist for vectored)?
            assert (
                len(last_rgb_array) == self.num_envs
            ), "First dimension (list) %d should equal number of environments %d" % (
                len(last_rgb_array),
                self.num_envs,
            )

            for index, sub_frame in zip(itertools.count(), last_rgb_array):
                _render_sub_frame(
                    sub_frame, subframe=(index // self._scale, index % self._scale)
                )
        else:
            _render_sub_frame(last_rgb_array)

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def _close(self):
        """Close the rendering window. Can be called even if the class is not fully instantiated."""
        if getattr(self, "window", None) is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
