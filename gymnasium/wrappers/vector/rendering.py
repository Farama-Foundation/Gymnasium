from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorWrapper, VectorEnv
from gymnasium.vector.vector_env import ArrayType


class HumanRendering(VectorWrapper):

    ACCEPTED_RENDER_MODES = [
        "rgb_array",
        "rgb_array_list",
        "depth_array",
        "depth_array_list"
    ]

    def __init__(self, env: VectorEnv, screen_size: tuple[int, int] | None = None):
        VectorWrapper.__init__(self, env)

        assert self.env.render_mode in self.ACCEPTED_RENDER_MODES, f"Expected env.render_mode to be one of {self.ACCEPTED_RENDER_MODES} but got '{env.render_mode}'"
        assert (
                "render_fps" in self.env.metadata
        ), "The base environment must specify 'render_fps' to be used with the HumanRendering wrapper"

        self.screen_size = screen_size
        self.window = None
        self.clock = None

        if "human" not in self.metadata["render_modes"]:
            self.metadata = deepcopy(self.env.metadata)
            self.metadata["render_modes"].append("human")


    @property
    def render_mode(self) -> str:
        """Always returns ``'human'``."""
        return "human"

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Perform a step in the base environment and render a frame to the screen."""
        result = super().step(actions)
        self._render_frame()
        return result

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the base environment and render a frame to the screen."""
        result = super().reset(seed=seed, options=options)
        self._render_frame()
        return result

    def render(self) -> None:
        """This method doesn't do much, actual rendering is performed in :meth:`step` and :meth:`reset`."""
        return None

    def _render_frame(self):
        """Fetch the last frame from the base environment and render it to the screen."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            )
        if self.env.render_mode.endswith("_last"):
            last_rgb_arrays = self.env.render()
            assert isinstance(last_rgb_arrays, list)
            last_rgb_arrays = last_rgb_arrays[-1]
        else:
            last_rgb_arrays = self.env.render()

        assert len(last_rgb_arrays) == self.num_envs
        assert all(isinstance(array, np.ndarray) for array in last_rgb_arrays), f'Expected `env.render()` to return a numpy array, actually returned {type(last_rgb_array)}'

        rgb_arrays = np.array(last_rgb_arrays, dtype=np.uint8)
        rgb_array = np.transpose(rgb_arrays, axes=(0, 2, 1, 3))

        if self.screen_size is None:
            self.screen_size = rgb_array.shape[:2]

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
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        """Close the rendering window."""
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
        super().close()