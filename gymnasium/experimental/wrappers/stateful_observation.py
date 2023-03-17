"""A collection of stateful observation wrappers.

* ``DelayObservationV0`` - A wrapper for delaying the returned observation
* ``TimeAwareObservationV0`` - A wrapper for adding time aware observations to environment observation
* ``FrameStackObservationV0`` - Frame stack the observations
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, SupportsFloat
from typing_extensions import Final

import numpy as np

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.experimental.vector.utils import (
    batch_space,
    concatenate,
    create_empty_array,
)
from gymnasium.experimental.wrappers.utils import create_zero_array
from gymnasium.spaces import Box, Dict, Tuple


class DelayObservationV0(
    gym.ObservationWrapper[ObsType, ActType, ObsType], gym.utils.RecordConstructorArgs
):
    """Wrapper which adds a delay to the returned observation.

    Before reaching the :attr:`delay` number of timesteps, returned observations is an array of zeros with
    the same shape as the observation space.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> env.reset(seed=123)
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})

        >>> env = DelayObservationV0(env, delay=2)
        >>> env.reset(seed=123)
        (array([0., 0., 0., 0.], dtype=float32), {})
        >>> env.step(env.action_space.sample())
        (array([0., 0., 0., 0.], dtype=float32), 1.0, False, False, {})
        >>> env.step(env.action_space.sample())
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), 1.0, False, False, {})

    Note:
        This does not support random delay values, if users are interested, please raise an issue or pull request to add this feature.
    """

    def __init__(self, env: gym.Env[ObsType, ActType], delay: int):
        """Initialises the DelayObservation wrapper with an integer.

        Args:
            env: The environment to wrap
            delay: The number of timesteps to delay observations
        """
        if not np.issubdtype(type(delay), np.integer):
            raise TypeError(
                f"The delay is expected to be an integer, actual type: {type(delay)}"
            )
        if not 0 <= delay:
            raise ValueError(
                f"The delay needs to be greater than zero, actual value: {delay}"
            )

        gym.utils.RecordConstructorArgs.__init__(self, delay=delay)
        gym.ObservationWrapper.__init__(self, env)

        self.delay: Final[int] = int(delay)
        self.observation_queue: Final[deque] = deque()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment, clearing the observation queue."""
        self.observation_queue.clear()

        return super().reset(seed=seed, options=options)

    def observation(self, observation: ObsType) -> ObsType:
        """Return the delayed observation."""
        self.observation_queue.append(observation)

        if len(self.observation_queue) > self.delay:
            return self.observation_queue.popleft()
        else:
            return create_zero_array(self.observation_space)


class TimeAwareObservationV0(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Augment the observation with time information of the episode.

    The :attr:`normalize_time` if ``True`` represents time as a normalized value between [0,1]
    otherwise if ``False``, the number of timesteps remaining before truncation occurs is an integer.

    For environments with ``Dict`` observation spaces, the time information is automatically
    added in the key `"time"` (can be changed through :attr:`dict_time_key`) and for environments with ``Tuple``
    observation space, the time information is added as the final element in the tuple.
    Otherwise, the observation space is transformed into a ``Dict`` observation space with two keys,
    `"obs"` for the base environment's observation and `"time"` for the time information.

    To flatten the observation, use the :attr:`flatten` parameter which will use the
    :func:`gymnasium.spaces.utils.flatten` function.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import TimeAwareObservationV0
        >>> env = gym.make("CartPole-v1")
        >>> env = TimeAwareObservationV0(env)
        >>> env.observation_space
        Dict('obs': Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32), 'time': Box(0.0, 1.0, (1,), float32))
        >>> env.reset(seed=42)[0]
        {'obs': array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), 'time': array([0.], dtype=float32)}
        >>> _ = env.action_space.seed(42)
        >>> env.step(env.action_space.sample())[0]
        {'obs': array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476], dtype=float32), 'time': array([0.002], dtype=float32)}

    Unnormalize time observation space example:
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservationV0(env, normalize_time=False)
        >>> env.observation_space
        Dict('obs': Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32), 'time': Box(0, 500, (1,), int32))
        >>> env.reset(seed=42)[0]
        {'obs': array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), 'time': array([500], dtype=int32)}
        >>> _ = env.action_space.seed(42)[0]
        >>> env.step(env.action_space.sample())[0]
        {'obs': array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476], dtype=float32), 'time': array([499], dtype=int32)}

    Flatten observation space example:
        >>> env = gym.make("CartPole-v1")
        >>> env = TimeAwareObservationV0(env, flatten=True)
        >>> env.observation_space
        Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38
          0.0000000e+00], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38 1.0000000e+00], (5,), float32)
        >>> env.reset(seed=42)[0]
        array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ,  0.        ],
              dtype=float32)
        >>> _ = env.action_space.seed(42)
        >>> env.step(env.action_space.sample())[0]
        array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476,  0.002     ],
              dtype=float32)
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
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
        gym.utils.RecordConstructorArgs.__init__(
            self,
            flatten=flatten,
            normalize_time=normalize_time,
            dict_time_key=dict_time_key,
        )
        gym.ObservationWrapper.__init__(self, env)

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

        self._timesteps: int = 0

        # Find the normalized time space
        if self.normalize_time:
            self._time_preprocess_func = lambda time: np.array(
                [time / self.max_timesteps], dtype=np.float32
            )
            time_space = Box(0.0, 1.0)
        else:
            self._time_preprocess_func = lambda time: np.array(
                [self.max_timesteps - time], dtype=np.int32
            )
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
            self.observation_space: gym.Space[WrapperObsType] = spaces.flatten_space(
                observation_space
            )
            self._obs_postprocess_func = lambda obs: spaces.flatten(
                observation_space, obs
            )
        else:
            self.observation_space: gym.Space[WrapperObsType] = observation_space
            self._obs_postprocess_func = lambda obs: obs

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Adds to the observation with the current time information.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time information appended to it
        """
        return self._obs_postprocess_func(
            self._append_data_func(
                observation, self._time_preprocess_func(self._timesteps)
            )
        )

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, incrementing the time step.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action with the next observation containing the timestep info
        """
        self._timesteps += 1

        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset the environment setting the time to zero.

        Args:
            seed: The seed to reset the environment
            options: The options used to reset the environment

        Returns:
            Resets the environment with the initial timestep info added the observation
        """
        self._timesteps = 0

        return super().reset(seed=seed, options=options)


class FrameStackObservationV0(
    gym.Wrapper[WrapperObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs,
):
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

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        stack_size: int,
        *,
        zeros_obs: ObsType | None = None,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env: The environment to apply the wrapper
            stack_size: The number of frames to stack with zero_obs being used originally.
            zeros_obs: Keyword only parameter that allows a custom padding observation at :meth:`reset`
        """
        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(
                f"The stack_size is expected to be an integer, actual type: {type(stack_size)}"
            )
        if not 1 < stack_size:
            raise ValueError(
                f"The stack_size needs to be greater than one, actual value: {stack_size}"
            )

        gym.utils.RecordConstructorArgs.__init__(self, stack_size=stack_size)
        gym.Wrapper.__init__(self, env)

        self.observation_space = batch_space(env.observation_space, n=stack_size)
        self.stack_size: Final[int] = stack_size

        self.zero_obs: Final[ObsType] = (
            zeros_obs if zeros_obs else create_zero_array(env.observation_space)
        )
        self._stacked_obs = deque(
            [self.zero_obs for _ in range(self.stack_size)], maxlen=self.stack_size
        )
        self._stacked_array = create_empty_array(
            env.observation_space, n=self.stack_size
        )

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
        self._stacked_obs.append(obs)

        updated_obs = deepcopy(
            concatenate(
                self.env.observation_space, self._stacked_obs, self._stacked_array
            )
        )
        return updated_obs, reward, terminated, truncated, info

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
        for _ in range(self.stack_size - 1):
            self._stacked_obs.append(self.zero_obs)
        self._stacked_obs.append(obs)

        updated_obs = deepcopy(
            concatenate(
                self.env.observation_space, self._stacked_obs, self._stacked_array
            )
        )
        return updated_obs, info
