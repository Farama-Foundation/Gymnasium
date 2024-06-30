"""File for rendering of vector-based environments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.vector.vector_env import ArrayType


class HumanRendering(VectorWrapper):
    """Adds support for Human-based Rendering for Vector-based environments."""

    ACCEPTED_RENDER_MODES = [
        "rgb_array",
        "rgb_array_list",
        "depth_array",
        "depth_array_list",
    ]

    def __init__(self, env: VectorEnv, screen_size: tuple[int, int] | None = None):
        """Constructor for Human Rendering of Vector-based environments.

        Args:
            env: The vector environment
            screen_size: The rendering screen size otherwise the environment sub-env render size is used
        """
        VectorWrapper.__init__(self, env)

        self.screen_size = screen_size
        self.scaled_subenv_size, self.num_rows, self.num_cols = None, None, None
        self.window = None  # Has to be initialized before asserts, as self.window is used in auto close
        self.clock = None

        assert (
            self.env.render_mode in self.ACCEPTED_RENDER_MODES
        ), f"Expected env.render_mode to be one of {self.ACCEPTED_RENDER_MODES} but got '{env.render_mode}'"
        assert (
            "render_fps" in self.env.metadata
        ), "The base environment must specify 'render_fps' to be used with the HumanRendering wrapper"

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

    def _render_frame(self):
        """Fetch the last frame from the base environment and render it to the screen."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            )

        assert self.env.render_mode is not None
        if self.env.render_mode.endswith("_last"):
            subenv_renders = self.env.render()
            assert isinstance(subenv_renders, list)
            subenv_renders = subenv_renders[-1]
        else:
            subenv_renders = self.env.render()

        assert subenv_renders is not None
        assert len(subenv_renders) == self.num_envs
        assert all(
            isinstance(render, np.ndarray) for render in subenv_renders
        ), f"Expected `env.render()` to return a numpy array, actually returned {[type(render) for render in subenv_renders]}"

        subenv_renders = np.array(subenv_renders, dtype=np.uint8)
        subenv_renders = np.transpose(subenv_renders, axes=(0, 2, 1, 3))
        # shape = (num envs, width, height, channels)

        if self.screen_size is None:
            self.screen_size = subenv_renders.shape[1:3]

        if self.scaled_subenv_size is None:
            subenv_size = subenv_renders.shape[1:3]
            width_ratio = subenv_size[0] / self.screen_size[0]
            height_ratio = subenv_size[1] / self.screen_size[1]

            num_rows, num_cols = 1, 1
            while num_rows * num_cols < self.num_envs:
                row_ratio = num_rows * height_ratio
                col_ratio = num_cols * width_ratio

                if row_ratio == col_ratio:
                    num_rows, num_cols = num_rows + 1, num_cols + 1
                elif row_ratio > col_ratio:
                    num_cols += 1
                else:
                    num_rows += 1

            scaling_factor = min(
                self.screen_size[0] / (num_cols * subenv_size[0]),
                self.screen_size[1] / (num_rows * subenv_size[1]),
            )
            assert (
                num_cols * subenv_size[0] * scaling_factor == self.screen_size[0]
            ) or (num_rows * subenv_size[1] * scaling_factor == self.screen_size[1])

            self.num_rows = num_rows
            self.num_cols = num_cols
            self.scaled_subenv_size = (
                int(subenv_size[0] * scaling_factor),
                int(subenv_size[1] * scaling_factor),
            )

            assert self.num_rows * self.num_cols >= self.num_envs
            assert self.scaled_subenv_size[0] * self.num_cols <= self.screen_size[0]
            assert self.scaled_subenv_size[1] * self.num_rows <= self.screen_size[1]

        # print(f'{self.num_envs=}, {self.num_rows=}, {self.num_cols=}, {self.screen_size=}, {self.scaled_subenv_size=}')

        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                'opencv (cv2) is not installed, run `pip install "gymnasium[other]"`'
            ) from e

        merged_rgb_array = np.zeros(self.screen_size + (3,), dtype=np.uint8)
        cols, rows = np.meshgrid(np.arange(self.num_cols), np.arange(self.num_rows))

        for i, col, row in zip(range(self.num_envs), cols.flatten(), rows.flatten()):
            scaled_render = cv2.resize(subenv_renders[i], self.scaled_subenv_size[::-1])
            x = col * self.scaled_subenv_size[0]
            y = row * self.scaled_subenv_size[1]

            merged_rgb_array[
                x : x + self.scaled_subenv_size[0],
                y : y + self.scaled_subenv_size[1],
            ] = scaled_render

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.screen_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(merged_rgb_array)
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
