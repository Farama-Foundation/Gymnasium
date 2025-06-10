"""Implementation of Atari 2600 Preprocessing following the guidelines of Machado et al., 2018."""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.spaces import Box


__all__ = ["AtariPreprocessing"]


class AtariPreprocessing(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Implements the common preprocessing techniques for Atari environments (excluding frame stacking).

    For frame stacking use :class:`gymnasium.wrappers.FrameStackObservation`.
    No vector version of the wrapper exists

    This class follows the guidelines in Machado et al. (2018),
    "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents".

    Specifically, the following preprocess stages applies to the atari environment:

    - Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
    - Frame skipping: The number of frames skipped between steps, 4 by default.
    - Max-pooling: Pools over the most recent two observations from the frame skips.
    - Termination signal when a life is lost: When the agent losses a life during the environment, then the environment is terminated.
        Turned off by default. Not recommended by Machado et al. (2018).
    - Resize to a square image: Resizes the atari environment original observation shape from 210x180 to 84x84 by default.
    - Grayscale observation: Makes the observation greyscale, enabled by default.
    - Grayscale new axis: Extends the last channel of the observation such that the image is 3-dimensional, not enabled by default.
    - Scale observation: Whether to scale the observation between [0, 1) or [0, 255), not scaled by default.

    Example:
        >>> import gymnasium as gym
        >>> import ale_py
        >>> gym.register_envs(ale_py)
        >>> env = gym.make("ALE/Pong-v5", frameskip=1)
        >>> env = AtariPreprocessing(
        ...     env,
        ...     noop_max=10, frame_skip=4, terminal_on_life_loss=True,
        ...     screen_size=84, grayscale_obs=False, grayscale_newaxis=False
        ... )

    Change logs:
     * Added in gym v0.12.2 (gym #1455)
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int | tuple[int, int] = 84,
        terminal_on_life_loss: bool = False,
        grayscale_obs: bool = True,
        grayscale_newaxis: bool = False,
        scale_obs: bool = False,
    ):
        """Wrapper for Atari 2600 preprocessing.

        Args:
            env (Env): The environment to apply the preprocessing
            noop_max (int): For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
            frame_skip (int): The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
            screen_size (int | tuple[int, int]): resize Atari frame.
            terminal_on_life_loss (bool): `if True`, then :meth:`step()` returns `terminated=True` whenever a
                life is lost.
            grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
                is returned.
            grayscale_newaxis (bool): `if True and grayscale_obs=True`, then a channel axis is added to
                grayscale observations to make them 3-dimensional.
            scale_obs (bool): if True, then observation normalized in range [0,1) is returned. It also limits memory
                optimization benefits of FrameStack Wrapper.

        Raises:
            DependencyNotInstalled: opencv-python package not installed
            ValueError: Disable frame-skipping in the original env
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=grayscale_obs,
            grayscale_newaxis=grayscale_newaxis,
            scale_obs=scale_obs,
        )
        gym.Wrapper.__init__(self, env)

        try:
            import cv2  # noqa: F401
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                'opencv-python package not installed, run `pip install "gymnasium[other]"` to get dependencies for atari'
            )

        assert frame_skip > 0
        assert (isinstance(screen_size, int) and screen_size > 0) or (
            isinstance(screen_size, tuple)
            and len(screen_size) == 2
            and all(isinstance(size, int) and size > 0 for size in screen_size)
        ), f"Expect the `screen_size` to be positive, actually: {screen_size}"
        if frame_skip > 1 and getattr(env.unwrapped, "_frameskip", None) != 1:
            raise ValueError(
                "Disable frame-skipping in the original env. Otherwise, more than one frame-skip will happen as through this wrapper"
            )
        assert noop_max >= 0
        self.noop_max = noop_max
        if noop_max > 0:
            assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.frame_skip = frame_skip
        self.screen_size: tuple[int, int] = (
            screen_size
            if isinstance(screen_size, tuple)
            else (screen_size, screen_size)
        )
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs

        # buffer of most recent two observations for max pooling
        assert isinstance(env.observation_space, Box)
        if grayscale_obs:
            self.obs_buffer = [
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            ]
        else:
            self.obs_buffer = [
                np.empty(env.observation_space.shape, dtype=np.uint8),
                np.empty(env.observation_space.shape, dtype=np.uint8),
            ]

        self.lives = 0
        self.game_over = False

        _low, _high, _dtype = (0, 1, np.float32) if scale_obs else (0, 255, np.uint8)
        _shape = (self.screen_size[1], self.screen_size[0], 1 if grayscale_obs else 3)
        if grayscale_obs and not grayscale_newaxis:
            _shape = _shape[:-1]  # Remove channel axis
        self.observation_space = Box(low=_low, high=_high, shape=_shape, dtype=_dtype)

    @property
    def ale(self):
        """Make ale as a class property to avoid serialization error."""
        return self.env.unwrapped.ale

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Applies the preprocessing for an :meth:`env.step`."""
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.frame_skip):
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self.game_over = terminated

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                terminated = terminated or new_lives < self.lives
                self.game_over = terminated
                self.lives = new_lives

            if terminated or truncated:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[1])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[1])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[0])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[0])
        return self._get_obs(), total_reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment using preprocessing."""
        # NoopReset
        _, reset_info = self.env.reset(seed=seed, options=options)

        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            _, _, terminated, truncated, step_info = self.env.step(0)
            reset_info.update(step_info)
            if terminated or truncated:
                _, reset_info = self.env.reset(seed=seed, options=options)

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)

        return self._get_obs(), reset_info

    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])

        import cv2

        obs = cv2.resize(
            self.obs_buffer[0],
            self.screen_size,
            interpolation=cv2.INTER_AREA,
        )

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs
