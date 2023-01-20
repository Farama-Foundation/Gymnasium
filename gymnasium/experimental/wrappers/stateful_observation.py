"""A collection of stateful observation wrappers.

* ``DelayObservationV0`` - A wrapper for delaying the returned observation
* ``TimeAwareObservationV0`` - A wrapper for adding time aware observations to environment observation
* ``FrameStackObservationV0`` - Frame stack the observations
* ``AtariPreprocessingV0`` - Preprocessing wrapper for atari environments
"""
from __future__ import annotations

from collections import deque
from typing import Any, SupportsFloat
from typing_extensions import Final


try:
    import jumpy as jp
except ImportError as e:
    raise ImportError("Jumpy is not installed, run `pip install jax-jumpy`") from e
import numpy as np

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium import Env
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete, Tuple
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate


class DelayObservationV0(gym.ObservationWrapper):
    """Wrapper which adds a delay to the returned observation."""

    def __init__(self, env: gym.Env, delay: int):
        """Initialize the DelayObservation wrapper.

        Args:
            env (Env): the wrapped environment
            delay (int): number of timesteps for delaying the observation.
                         Before reaching the `delay` number of timesteps,
                         returned observation is an array of zeros with the
                         same shape of the observation space.
        """
        assert isinstance(
            env.observation_space, (Box, MultiBinary, MultiDiscrete)
        ), type(env.observation_space)
        assert 0 < delay

        self.delay: Final[int] = delay
        self.observation_queue: Final[deque] = deque()

        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment, clearing the observation queue."""
        self.observation_queue.clear()

        return super().reset(seed=seed, options=options)

    def observation(self, observation: ObsType) -> ObsType:
        """Return the delayed observation."""
        self.observation_queue.append(observation)

        if len(self.observation_queue) > self.delay:
            return self.observation_queue.popleft()

        return jp.zeros_like(observation)


class TimeAwareObservationV0(gym.ObservationWrapper):
    """Augment the observation with time information of the episode.

    Time can be represented as a normalized value between [0,1]
    or by the number of timesteps remaining before truncation occurs.

    For environments with ``Dict`` or ``Tuple`` observation spaces, by default,
    the time information is automatically added in the key `"time"` and
    as the final element in the tuple.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import TimeAwareObservationV0
        >>> env = gym.make("CartPole-v1")
        >>> env = TimeAwareObservationV0(env)
        >>> env.observation_space
        Dict('obs': Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32), 'time': Box(0.0, 1.0, (1,), float32))
        >>> _ = env.reset(seed=42)
        >>> _ = env.action_space.seed(42)
        >>> env.step(env.action_space.sample())[0]
        {'obs': array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476], dtype=float32), 'time': 0.002}

    Flatten observation space example:
        >>> env = gym.make("CartPole-v1")
        >>> env = TimeAwareObservationV0(env, flatten=True)
        >>> env.observation_space
        Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38
          0.0000000e+00], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38 1.0000000e+00], (5,), float32)
        >>> _ = env.reset(seed=42)
        >>> _ = env.action_space.seed(42)
        >>> env.step(env.action_space.sample())[0]
        array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476,  0.002     ],
              dtype=float32)

    """

    def __init__(
        self,
        env: gym.Env,
        flatten: bool = False,
        normalize_time: bool = True,
        *,
        dict_time_key: str = "time",
    ):
        """Initialize :class:`TimeAwareObservationV0`.

        Args:
            env: The environment to apply the wrapper
            flatten: Flatten the observation to a `Box` of a single dimension
            normalize_time: if `True` return time in the range [0,1]
                otherwise return time as remaining timesteps before truncation
            dict_time_key: For environment with a ``Dict`` observation space, the key for the time space. By default, `"time"`.
        """
        super().__init__(env)
        self.flatten: Final[bool] = flatten
        self.normalize_time: Final[bool] = normalize_time

        if hasattr(env, "_max_episode_steps"):
            self.max_timesteps = getattr(env, "_max_episode_steps")
        elif env.spec is not None and env.spec.max_episode_steps is not None:
            self.max_timesteps = env.spec.max_episode_steps
        else:
            raise ValueError(
                "The environment must be wrapped by a TimeLimit wrapper or the spec specify a `max_episode_steps`."
            )

        self.timesteps: int = 0

        # Find the normalized time space
        if self.normalize_time:
            self._time_preprocess_func = lambda time: time / self.max_timesteps
            time_space = Box(0.0, 1.0)
        else:
            self._time_preprocess_func = lambda time: self.max_timesteps - time
            time_space = Box(0, self.max_timesteps, dtype=np.int32)

        # Find the observation space
        if isinstance(env.observation_space, Dict):
            assert dict_time_key not in env.observation_space.keys()
            observation_space = Dict(
                {dict_time_key: time_space, **env.observation_space.spaces}
            )
            self._append_data_func = lambda obs, time: {dict_time_key: time, **obs}
        elif isinstance(env.observation_space, Tuple):
            observation_space = Tuple(env.observation_space.spaces + (time_space,))
            self._append_data_func = lambda obs, time: obs + (time,)
        else:
            observation_space = Dict(obs=env.observation_space, time=time_space)
            self._append_data_func = lambda obs, time: {"obs": obs, "time": time}

        # If to flatten the observation space
        if self.flatten:
            self.observation_space = spaces.flatten_space(observation_space)
            self._obs_postprocess_func = lambda obs: spaces.flatten(
                observation_space, obs
            )
        else:
            self.observation_space = observation_space
            self._obs_postprocess_func = lambda obs: obs

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Adds to the observation with the current time information.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time information appended to
        """
        return self._obs_postprocess_func(
            self._append_data_func(
                observation, self._time_preprocess_func(self.timesteps)
            )
        )

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, incrementing the time step.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        self.timesteps += 1
        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset the environment setting the time to zero.

        Args:
            seed: The seed to reset the environment
            options: The options used to reset the environment

        Returns:
            The reset environment
        """
        self.timesteps = 0

        return super().reset(seed=seed, options=options)


class FrameStackObservationV0(gym.Wrapper):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - After :meth:`reset` is called, the frame buffer will be filled with the initial observation.
          I.e. the observation returned by :meth:`reset` will consist of `num_stack` many identical frames.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import FrameStackObservationV0
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStackObservationV0(env, 4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(self, env: Env[ObsType, ActType], stack_size: int):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env: The environment to apply the wrapper
            stack_size: The number of frames to stack
        """
        assert np.issubdtype(type(stack_size), np.integer)
        assert stack_size > 0

        super().__init__(env)

        self.observation_space = batch_space(env.observation_space, n=stack_size)
        self.stack_size = stack_size

        self.stacked_obs_array = create_empty_array(env.observation_space, n=stack_size)
        self.stacked_obs = self._init_stacked_obs()

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and info from the environment
        """
        obs, reward, terminated, truncated, info = super().step(action)
        self.stacked_obs.rotate(1)
        self.stacked_obs[0] = obs

        return (
            concatenate(
                self.observation_space, self.stacked_obs, self.stacked_obs_array
            ),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset the environment, returning the stacked observation and info.

        Args:
            seed: The environment seed
            options: The reset options

        Returns:
            The stacked observations and info
        """
        obs, info = super().reset(seed=seed, options=options)
        self.stacked_obs = self._init_stacked_obs()
        self.stacked_obs[0] = obs

        return (
            concatenate(
                self.observation_space, self.stacked_obs, self.stacked_obs_array
            ),
            info,
        )

    def _init_stacked_obs(self) -> deque:
        return deque(
            iterate(
                self.observation_space,
                create_empty_array(self.env.observation_space, n=self.stack_size),
            )
        )
