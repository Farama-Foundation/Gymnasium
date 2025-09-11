"""File for rendering and recording of vector-based environments."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium import error, logger
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.error import DependencyNotInstalled
from gymnasium.logger import warn
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.vector.vector_env import ArrayType


class HumanRendering(VectorWrapper, gym.utils.RecordConstructorArgs):
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
        gym.utils.RecordConstructorArgs.__init__(self, screen_size=screen_size)

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


class RecordVideo(
    gym.vector.VectorWrapper,
    gym.utils.RecordConstructorArgs,
):
    """Adds support for video recording for Vector-based environments.

    The class is the same as gymnasium.wrappers.rendering.RecordVideo, but
    expects multiple frames when rendering the environment (one for each
    environment of the VectorEnv). Frames are concatenated into one frame such
    that its aspect ratio is as close as possible to the desired one.

    Examples - Run 5 environments for 200 timesteps, and save the video every 5 episodes:
    >>> import os
    >>> import gymnasium as gym
    >>> from gymnasium.wrappers.vector import RecordVideo
    >>> envs = gym.make_vec("CartPole-v1", num_envs=5, render_mode="rgb_array")
    >>> envs = RecordVideo(
    ...     envs,
    ...     video_folder="save_videos_5envs",
    ...     video_aspect_ratio=(1,1),
    ...     episode_trigger=lambda t: t % 5 == 0,
    ... )
    >>> _ = envs.reset(seed=123)
    >>> _ = envs.action_space.seed(123)
    >>> for i in range(200):
    ...     actions = envs.action_space.sample()
    ...     _ = envs.step(actions)
    >>> envs.close()
    >>> len(os.listdir("save_videos_5envs"))
    2
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        video_folder: str,
        video_aspect_ratio: tuple[int, int] = (1, 1),
        record_first_only: bool = False,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
    ):
        """Wrapper records videos of environment rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            video_aspect_ratio (tuple): the desired aspect ratio of the video concatenating all environments. For example, (1, 1) means an
                aspect ratio of 1:1, while (16, 9) means 16:9.
            record_first_only (bool): if True, only the first environment is recorded.
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            fps (int): The frame per second in the video. Provides a custom video fps for environment, if ``None`` then
                the environment metadata ``render_fps`` key is used if it exists, otherwise a default value of 30 is used.
            disable_logger (bool): Whether to disable moviepy logger or not, default it is disabled
            gc_trigger: Function that accepts an integer and returns ``True`` iff garbage collection should be performed after this episode

        Note:
            For vector environments that use same-step autoreset (see https://farama.org/Vector-Autoreset-Mode for more details)
            then the final frame of the episode will not be included in the video.
        """
        VectorWrapper.__init__(self, env)
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )

        if env.render_mode in {None, "human", "ansi"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordVideo.",
                "Initialize your environment with a render_mode that returns an image, such as rgb_array.",
            )

        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule

            episode_trigger = capped_cubic_video_schedule

        autoreset_mode = env.metadata.get("autoreset_mode", None)
        if autoreset_mode is not None:
            self.autoreset_mode = autoreset_mode
        else:
            warn(
                f"{env} metadata doesn't specify its autoreset mode ({env.metadata!r}), therefore, defaulting to next step."
            )
            self.autoreset_mode = gym.vector.AutoresetMode.NEXT_STEP
        if self.autoreset_mode == gym.vector.AutoresetMode.SAME_STEP:
            logger.warn(
                "Vector environment's autoreset mode is same-step (https://farama.org/Vector-Autoreset-Mode). Recorded episodes will not contain the last frame of the episode."
            )
        self.has_autoreset = False

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger
        self.gc_trigger = gc_trigger

        self.record_first_only = record_first_only
        self.video_aspect_ratio = video_aspect_ratio
        self.frame_cols = -1
        self.frame_rows = -1

        self.video_folder = os.path.abspath(video_folder)
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        if fps is None:
            fps = self.metadata.get("render_fps", 30)
        self.frames_per_sec: int = fps
        self.name_prefix: str = name_prefix
        self._video_name: str | None = None
        self.video_length: int = video_length if video_length != 0 else float("inf")
        self.recording: bool = False
        self.recorded_frames: list[np.ndarray] = []
        self.render_history: list[np.ndarray] = []

        self.step_id = -1
        self.episode_id = -1

        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise error.DependencyNotInstalled(
                'MoviePy is not installed, run `pip install "gymnasium[other]"`'
            ) from e

    def _get_concat_frame_shape(self, n_frames, h, w):
        """Finds the right shape to concatenate frames from all environments into one frame."""
        target_video_aspect_ratio = (
            self.video_aspect_ratio[0] / self.video_aspect_ratio[1]
        )
        best_rows, best_cols = 1, n_frames
        min_diff = np.inf
        for rows in range(1, int(n_frames**0.5) + 1):
            if n_frames % rows == 0:
                cols = n_frames // rows
                total_height = rows * h
                total_width = cols * w
                aspect = total_width / total_height
                diff = abs(aspect - target_video_aspect_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_rows, best_cols = rows, cols
        self.frame_rows = best_rows
        self.frame_cols = best_cols

    def _concat_frames(self, frames):
        """Concatenates a list of frames into one large frame."""
        frames = np.array(frames)
        n_frames, h, w, c = frames.shape
        grid = np.zeros(
            (self.frame_rows * h, self.frame_cols * w, c), dtype=frames.dtype
        )
        for idx in range(n_frames):
            r = idx // self.frame_cols
            c_ = idx % self.frame_cols
            grid[r * h : (r + 1) * h, c_ * w : (c_ + 1) * w] = frames[idx]
        return grid

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        envs_frame = self.env.render()
        assert isinstance(envs_frame, Sequence), type(envs_frame)
        assert len(envs_frame) == self.num_envs

        if self.record_first_only:
            envs_frame = [envs_frame[0]]

        if self.frame_cols == -1 or self.frame_rows == -1:
            n_frames = len(envs_frame)
            h, w, c = envs_frame[0].shape
            self._get_concat_frame_shape(n_frames, h, w)

        concatenated_envs_frame = self._concat_frames(envs_frame)
        self.recorded_frames.append(concatenated_envs_frame)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        if options is None or "reset_mask" not in options or options["reset_mask"][0]:
            self.episode_id += 1

            if self.recording and self.video_length == float("inf"):
                self.stop_recording()

            if self.episode_trigger and self.episode_trigger(self.episode_id):
                self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")

        obs, info = super().reset(seed=seed, options=options)

        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        self.has_autoreset = False

        return obs, info

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        obs, rewards, terminations, truncations, info = self.env.step(actions)
        self.step_id += 1

        if self.autoreset_mode == gym.vector.AutoresetMode.NEXT_STEP:
            if self.has_autoreset:
                self.episode_id += 1
                if self.recording and self.video_length == float("inf"):
                    self.stop_recording()

                if self.episode_trigger and self.episode_trigger(self.episode_id):
                    self.start_recording(
                        f"{self.name_prefix}-episode-{self.episode_id}"
                    )
            self.has_autoreset = terminations[0] or truncations[0]
        elif self.autoreset_mode == gym.vector.AutoresetMode.SAME_STEP and (
            terminations[0] or truncations[0]
        ):
            self.episode_id += 1
            if self.recording and self.video_length == float("inf"):
                self.stop_recording()

            if self.episode_trigger and self.episode_trigger(self.episode_id):
                self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")

        if self.recording:
            self._capture_frame()

            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return obs, rewards, terminations, truncations, info

    def render(self) -> RenderFrame | list[RenderFrame]:
        """Compute the render frames as specified by render_mode attribute during initialization of the environment."""
        render_out = super().render()
        if self.recording and isinstance(render_out, list):
            self.recorded_frames += render_out

        if len(self.render_history) > 0:
            tmp_history = self.render_history
            self.render_history = []
            return tmp_history + render_out
        else:
            return render_out

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self, video_name: str):
        """Start a new recording. If it is already recording, stops the current recording before starting the new one."""
        if self.recording:
            self.stop_recording()

        self.recording = True
        self._video_name = video_name

    def stop_recording(self):
        """Stop current recording and saves the video."""
        assert self.recording, "stop_recording was called, but no recording was started"
        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            clip.write_videofile(path, logger=moviepy_logger)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None

        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()

    def __del__(self):
        """Warn the user in case last video wasn't saved."""
        if len(self.recorded_frames) > 0:
            logger.warn("Unable to save last video! Did you call close()?")
