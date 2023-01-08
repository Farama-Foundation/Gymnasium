"""A collections of rendering-based wrappers.

* ``RenderCollectionV0`` - Collects rendered frames into a list
* ``RecordVideoV0`` - Records a video of the environments
* ``HumanRenderingV0`` - Provides human rendering of environments with ``"rgb_array"``
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame, WrapperActType, WrapperObsType
from gymnasium.error import DependencyNotInstalled


class RenderCollectionV0(gym.Wrapper):
    """Collect rendered frames of an environment such ``render`` returns a ``list[RenderedFrame]``."""

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        pop_frames: bool = True,
        reset_clean: bool = True,
    ):
        """Initialize a :class:`RenderCollection` instance.

        Args:
            env: The environment that is being wrapped
            pop_frames (bool): If true, clear the collection frames after ``meth:render`` is called. Default value is ``True``.
            reset_clean (bool): If true, clear the collection frames when ``meth:reset`` is called. Default value is ``True``.
        """
        super().__init__(env)
        assert env.render_mode is not None
        assert not env.render_mode.endswith("_list")

        self.frame_list: list[RenderFrame] = []
        self.pop_frames = pop_frames
        self.reset_clean = reset_clean

        self.metadata = deepcopy(self.env.metadata)
        if f"{self.env.render_mode}_list" not in self.metadata["render_modes"]:
            self.metadata["render_modes"].append(f"{self.env.render_mode}_list")

    @property
    def render_mode(self):
        """Returns the collection render_mode name."""
        return f"{self.env.render_mode}_list"

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Perform a step in the base environment and collect a frame."""
        output = super().step(action)
        self.frame_list.append(super().render())
        return output

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset the base environment, eventually clear the frame_list, and collect a frame."""
        output = super().reset(seed=seed, options=options)

        if self.reset_clean:
            self.frame_list = []
        self.frame_list.append(super().render())

        return output

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Returns the collection of frames and, if pop_frames = True, clears it."""
        frames = self.frame_list
        if self.pop_frames:
            self.frame_list = []

        return frames


class RecordVideoV0(gym.Wrapper):
    """Record a video of an environment."""

    pass


class HumanRenderingV0(gym.Wrapper):
    """Performs human rendering for an environment that only supports "rgb_array"rendering.

    This wrapper is particularly useful when you have implemented an environment that can produce
    RGB images but haven't implemented any code to render the images to the screen.
    If you want to use this wrapper with your environments, remember to specify ``"render_fps"``
    in the metadata of your environment.

    The ``render_mode`` of the wrapped environment must be either ``'rgb_array'`` or ``'rgb_array_list'``.

    Example:
        >>> env = gym.make("LunarLander-v2", render_mode="rgb_array")
        >>> wrapped = HumanRenderingV0(env)
        >>> wrapped.reset()     # This will start rendering to the screen

    The wrapper can also be applied directly when the environment is instantiated, simply by passing
    ``render_mode="human"`` to ``make``. The wrapper will only be applied if the environment does not
    implement human-rendering natively (i.e. ``render_mode`` does not contain ``"human"``).

    Example:
        >>> env = gym.make("NoNativeRendering-v2", render_mode="human")      # NoNativeRendering-v0 doesn't implement human-rendering natively
        >>> env.reset()     # This will start rendering to the screen

    Warning: If the base environment uses ``render_mode="rgb_array_list"``, its (i.e. the *base environment's*) render method
        will always return an empty list:

            >>> env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
            >>> wrapped = HumanRenderingV0(env)
            >>> wrapped.reset()
            >>> env.render()
            []          # env.render() will always return an empty list!

    """

    def __init__(self, env):
        """Initialize a :class:`HumanRendering` instance.

        Args:
            env: The environment that is being wrapped
        """
        super().__init__(env)
        assert env.render_mode in [
            "rgb_array",
            "rgb_array_list",
        ], f"Expected env.render_mode to be one of 'rgb_array' or 'rgb_array_list' but got '{env.render_mode}'"
        assert (
            "render_fps" in env.metadata
        ), "The base environment must specify 'render_fps' to be used with the HumanRendering wrapper"

        self.screen_size = None
        self.window = None
        self.clock = None

        if "human" not in self.metadata["render_modes"]:
            self.metadata = deepcopy(self.env.metadata)
            self.metadata["render_modes"].append("human")

    @property
    def render_mode(self):
        """Always returns ``'human'``."""
        return "human"

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Perform a step in the base environment and render a frame to the screen."""
        result = super().step(action)
        self._render_frame()
        return result

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset the base environment and render a frame to the screen."""
        result = super().reset(seed=seed, options=options)
        self._render_frame()
        return result

    def render(self):
        """This method doesn't do much, actual rendering is performed in :meth:`step` and :meth:`reset`."""
        return None

    def _render_frame(self):
        """Fetch the last frame from the base environment and render it to the screen."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[box2d]`"
            )
        if self.env.render_mode == "rgb_array_list":
            last_rgb_array = self.env.render()
            assert isinstance(last_rgb_array, list)
            last_rgb_array = last_rgb_array[-1]
        elif self.env.render_mode == "rgb_array":
            last_rgb_array = self.env.render()
        else:
            raise Exception(
                f"Wrapped environment must have mode 'rgb_array' or 'rgb_array_list', actual render mode: {self.env.render_mode}"
            )
        assert isinstance(last_rgb_array, np.ndarray)

        rgb_array = np.transpose(last_rgb_array, axes=(1, 0, 2))

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
