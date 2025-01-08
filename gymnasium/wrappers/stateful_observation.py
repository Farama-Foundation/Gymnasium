"""A collection of stateful observation wrappers.

* ``DelayObservation`` - A wrapper for delaying the returned observation
* ``TimeAwareObservation`` - A wrapper for adding time aware observations to environment observation
* ``FrameStackObservation`` - Frame stack the observations
* ``NormalizeObservation`` - Normalized the observations to have unit variance with a moving mean
* ``MaxAndSkipObservation`` - Return only every ``skip``-th frame (frameskipping) and return the max between the two last frames.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Final, SupportsFloat

import numpy as np

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.spaces import Box, Dict, Tuple
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from gymnasium.wrappers.utils import RunningMeanStd, create_zero_array


__all__ = [
    "DelayObservation",
    "TimeAwareObservation",
    "FrameStackObservation",
    "NormalizeObservation",
    "MaxAndSkipObservation",
]


class DelayObservation(
    gym.ObservationWrapper[ObsType, ActType, ObsType], gym.utils.RecordConstructorArgs
):
    """Adds a delay to the returned observation from the environment.

    Before reaching the :attr:`delay` number of timesteps, returned observations is an array of zeros with
    the same shape as the observation space.

    No vector version of the wrapper exists.

    Note:
        This does not support random delay values, if users are interested, please raise an issue or pull request to add this feature.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> env.reset(seed=123)
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})

        >>> env = DelayObservation(env, delay=2)
        >>> env.reset(seed=123)
        (array([0., 0., 0., 0.], dtype=float32), {})
        >>> env.step(env.action_space.sample())
        (array([0., 0., 0., 0.], dtype=float32), 1.0, False, False, {})
        >>> env.step(env.action_space.sample())
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), 1.0, False, False, {})

    Change logs:
     * v1.0.0 - Initially added
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


class TimeAwareObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Augment the observation with the number of time steps taken within an episode.

    The :attr:`normalize_time` if ``True`` represents time as a normalized value between [0,1]
    otherwise if ``False``, the current timestep is an integer.

    For environments with ``Dict`` observation spaces, the time information is automatically
    added in the key `"time"` (can be changed through :attr:`dict_time_key`) and for environments with ``Tuple``
    observation space, the time information is added as the final element in the tuple.
    Otherwise, the observation space is transformed into a ``Dict`` observation space with two keys,
    `"obs"` for the base environment's observation and `"time"` for the time information.

    To flatten the observation, use the :attr:`flatten` parameter which will use the
    :func:`gymnasium.spaces.utils.flatten` function.

    No vector version of the wrapper exists.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TimeAwareObservation
        >>> env = gym.make("CartPole-v1")
        >>> env = TimeAwareObservation(env)
        >>> env.observation_space
        Box([-4.80000019        -inf -0.41887903        -inf  0.        ], [4.80000019e+00            inf 4.18879032e-01            inf
         5.00000000e+02], (5,), float64)
        >>> env.reset(seed=42)[0]
        array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ,  0.        ])
        >>> _ = env.action_space.seed(42)
        >>> env.step(env.action_space.sample())[0]
        array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476,  1.        ])

    Normalize time observation space example:
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservation(env, normalize_time=True)
        >>> env.observation_space
        Box([-4.8               -inf -0.41887903        -inf  0.        ], [4.8               inf 0.41887903        inf 1.        ], (5,), float32)
        >>> env.reset(seed=42)[0]
        array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ,  0.        ],
              dtype=float32)
        >>> _ = env.action_space.seed(42)
        >>> env.step(env.action_space.sample())[0]
        array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476,  0.002     ],
              dtype=float32)

    Flatten observation space example:
        >>> env = gym.make("CartPole-v1")
        >>> env = TimeAwareObservation(env, flatten=False)
        >>> env.observation_space
        Dict('obs': Box([-4.8               -inf -0.41887903        -inf], [4.8               inf 0.41887903        inf], (4,), float32), 'time': Box(0, 500, (1,), int32))
        >>> env.reset(seed=42)[0]
        {'obs': array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), 'time': array([0], dtype=int32)}
        >>> _ = env.action_space.seed(42)
        >>> env.step(env.action_space.sample())[0]
        {'obs': array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476], dtype=float32), 'time': array([1], dtype=int32)}

    Change logs:
     * v0.18.0 - Initially added
     * v1.0.0 - Remove vector environment support, add ``flatten`` and ``normalize_time`` parameters
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        flatten: bool = True,
        normalize_time: bool = False,
        *,
        dict_time_key: str = "time",
    ):
        """Initialize :class:`TimeAwareObservation`.

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

        # We don't need to keep if a TimeLimit wrapper exists as `spec` will do that work for us now
        if env.spec is not None and env.spec.max_episode_steps is not None:
            self.max_timesteps = env.spec.max_episode_steps
        else:
            # else we need to loop through the environment stack to check if a `TimeLimit` wrapper exists
            wrapped_env = env
            while isinstance(wrapped_env, gym.Wrapper):
                if isinstance(wrapped_env, gym.wrappers.TimeLimit):
                    self.max_timesteps = wrapped_env._max_episode_steps
                    break
                wrapped_env = wrapped_env.env

            if not isinstance(wrapped_env, gym.wrappers.TimeLimit):
                raise ValueError(
                    "The environment must be wrapped by a TimeLimit wrapper or the spec specify a `max_episode_steps`."
                )

        self.timesteps: int = 0

        # Find the normalized time space
        if self.normalize_time:
            self._time_preprocess_func = lambda time: np.array(
                [time / self.max_timesteps], dtype=np.float32
            )
            time_space = Box(0.0, 1.0)
        else:
            self._time_preprocess_func = lambda time: np.array([time], dtype=np.int32)
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
            The environment's step using the action with the next observation containing the timestep info
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
            Resets the environment with the initial timestep info added the observation
        """
        self.timesteps = 0

        return super().reset(seed=seed, options=options)


class FrameStackObservation(
    gym.Wrapper[WrapperObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Stacks the observations from the last ``N`` time steps in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Users have options for the padded observation used:

     * "reset" (default) - The reset value is repeated
     * "zero" - A "zero"-like instance of the observation space
     * custom - An instance of the observation space

    No vector version of the wrapper exists.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStackObservation
        >>> env = gym.make("CarRacing-v3")
        >>> env = FrameStackObservation(env, stack_size=4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)

    Example with different padding observations:
        >>> env = gym.make("CartPole-v1")
        >>> env.reset(seed=123)
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})
        >>> stacked_env = FrameStackObservation(env, 3)   # the default is padding_type="reset"
        >>> stacked_env.reset(seed=123)
        (array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32), {})


        >>> stacked_env = FrameStackObservation(env, 3, padding_type="zero")
        >>> stacked_env.reset(seed=123)
        (array([[ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32), {})
        >>> stacked_env = FrameStackObservation(env, 3, padding_type=np.array([1, -1, 0, 2], dtype=np.float32))
        >>> stacked_env.reset(seed=123)
        (array([[ 1.        , -1.        ,  0.        ,  2.        ],
               [ 1.        , -1.        ,  0.        ,  2.        ],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32), {})

    Change logs:
     * v0.15.0 - Initially add as ``FrameStack`` with support for lz4
     * v1.0.0 - Rename to ``FrameStackObservation`` and remove lz4 and ``LazyFrame`` support
                along with adding the ``padding_type`` parameter

    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        stack_size: int,
        *,
        padding_type: str | ObsType = "reset",
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env: The environment to apply the wrapper
            stack_size: The number of frames to stack.
            padding_type: The padding type to use when stacking the observations, options: "reset", "zero", custom obs
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, stack_size=stack_size, padding_type=padding_type
        )
        gym.Wrapper.__init__(self, env)

        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(
                f"The stack_size is expected to be an integer, actual type: {type(stack_size)}"
            )
        if not 0 < stack_size:
            raise ValueError(
                f"The stack_size needs to be greater than zero, actual value: {stack_size}"
            )
        if isinstance(padding_type, str) and (
            padding_type == "reset" or padding_type == "zero"
        ):
            self.padding_value: ObsType = create_zero_array(env.observation_space)
        elif padding_type in env.observation_space:
            self.padding_value = padding_type
            padding_type = "_custom"
        else:
            if isinstance(padding_type, str):
                raise ValueError(  # we are guessing that the user just entered the "reset" or "zero" wrong
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r}"
                )
            else:
                raise ValueError(
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r} not an instance of env observation ({env.observation_space})"
                )

        self.observation_space = batch_space(env.observation_space, n=stack_size)
        self.stack_size: Final[int] = stack_size
        self.padding_type: Final[str] = padding_type

        self.obs_queue = deque(
            [self.padding_value for _ in range(self.stack_size)], maxlen=self.stack_size
        )
        self.stacked_obs = create_empty_array(env.observation_space, n=self.stack_size)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and info from the environment
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.append(obs)

        updated_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
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
        obs, info = self.env.reset(seed=seed, options=options)

        if self.padding_type == "reset":
            self.padding_value = obs
        for _ in range(self.stack_size - 1):
            self.obs_queue.append(self.padding_value)
        self.obs_queue.append(obs)

        updated_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
        )
        return updated_obs, info


class NormalizeObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Normalizes observations to be centered at the mean with unit variance.

    The property :attr:`update_running_mean` allows to freeze/continue the running mean calculation of the observation
    statistics. If ``True`` (default), the ``RunningMeanStd`` will get updated every time ``step`` or ``reset`` is called.
    If ``False``, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.NormalizeObservation`.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.

    Example:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> obs, info = env.reset(seed=123)
        >>> term, trunc = False, False
        >>> while not (term or trunc):
        ...     obs, _, term, trunc, _ = env.step(1)
        ...
        >>> obs
        array([ 0.1511158 ,  1.7183299 , -0.25533703, -2.8914354 ], dtype=float32)
        >>> env = gym.make("CartPole-v1")
        >>> env = NormalizeObservation(env)
        >>> obs, info = env.reset(seed=123)
        >>> term, trunc = False, False
        >>> while not (term or trunc):
        ...     obs, _, term, trunc, _ = env.step(1)
        >>> obs
        array([ 2.0059888,  1.5676788, -1.9944268, -1.6120394], dtype=float32)

    Change logs:
     * v0.21.0 - Initially add
     * v1.0.0 - Add `update_running_mean` attribute to allow disabling of updating the running mean / standard, particularly useful for evaluation time.
        Casts all observations to `np.float32` and sets the observation space with low/high of `-np.inf` and `np.inf` and dtype as `np.float32`
    """

    def __init__(self, env: gym.Env[ObsType, ActType], epsilon: float = 1e-8):
        """This wrapper will normalize observations such that each observation is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.ObservationWrapper.__init__(self, env)

        assert env.observation_space.shape is not None
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

        self.obs_rms = RunningMeanStd(
            shape=self.observation_space.shape, dtype=self.observation_space.dtype
        )
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Normalises the observation using the running mean and variance of the observations."""
        if self._update_running_mean:
            self.obs_rms.update(np.array([observation]))
        return np.float32(
            (observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        )


class MaxAndSkipObservation(
    gym.Wrapper[WrapperObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Skips the N-th frame (observation) and return the max values between the two last observations.

    No vector version of the wrapper exists.

    Note:
        This wrapper is based on the wrapper from [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#MaxAndSkipEnv)

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> obs0, *_ = env.reset(seed=123)
        >>> obs1, *_ = env.step(1)
        >>> obs2, *_ = env.step(1)
        >>> obs3, *_ = env.step(1)
        >>> obs4, *_ = env.step(1)
        >>> skip_and_max_obs = np.max(np.stack([obs3, obs4], axis=0), axis=0)
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = MaxAndSkipObservation(env)
        >>> wrapped_obs0, *_ = wrapped_env.reset(seed=123)
        >>> wrapped_obs1, *_ = wrapped_env.step(1)
        >>> np.all(obs0 == wrapped_obs0)
        np.True_
        >>> np.all(wrapped_obs1 == skip_and_max_obs)
        np.True_

    Change logs:
     * v1.0.0 - Initially add
    """

    def __init__(self, env: gym.Env[ObsType, ActType], skip: int = 4):
        """This wrapper will return only every ``skip``-th frame (frameskipping) and return the max between the two last frames.

        Args:
            env (Env): The environment to apply the wrapper
            skip: The number of frames to skip
        """
        gym.utils.RecordConstructorArgs.__init__(self, skip=skip)
        gym.Wrapper.__init__(self, env)

        if not np.issubdtype(type(skip), np.integer):
            raise TypeError(
                f"The skip is expected to be an integer, actual type: {type(skip)}"
            )
        if skip < 2:
            raise ValueError(
                f"The skip value needs to be equal or greater than two, actual value: {skip}"
            )
        if env.observation_space.shape is None:
            raise ValueError("The observation space must have the shape attribute.")

        self._skip = skip
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment with the given action for ``skip`` steps.

        Repeat action, sum reward, and max over last observations.

        Args:
            action: The action to step through the environment with
        Returns:
            Max of the last two observations, reward, terminated, truncated, and info from the environment
        """
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = np.max(self._obs_buffer, axis=0)

        return max_frame, total_reward, terminated, truncated, info
