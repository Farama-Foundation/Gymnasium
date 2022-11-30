"""Wrapper for adding time aware observations to environment observation."""
from collections import OrderedDict

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Dict


class TimeAwareObservationV0(gym.ObservationWrapper):
    """Augment the observation with time information of the episode.

    Time can be represented as a normalized value between [0,1]
    or by the number of timesteps remaining before truncation occurs.

    Example:
        >>> import gym
        >>> from gym.wrappers import TimeAwareObservationV0
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservationV0(env)
        >>> env.observation_space
        Dict(obs: Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32), time: Box(0.0, 500, (1,), float32))
        >>> _ = env.reset()
        >>> env.step(env.action_space.sample())[0]
        OrderedDict([('obs',
        ...       array([ 0.02866629,  0.2310988 , -0.02614601, -0.2600732 ], dtype=float32)),
        ...      ('time', array([0.002]))])

    Flatten observation space example:
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservationV0(env, flatten=True)
        >>> env.observation_space
        Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38  0.0000000e+00], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38 500], (5,), float32)
        >>> _ = env.reset()
        >>> env.step(env.action_space.sample())[0]
        array([-0.01232257,  0.19335455, -0.02244143, -0.32388705,  0.002 ], dtype=float32)
    """

    def __init__(self, env: gym.Env, flatten=False, normalize_time=True):
        """Initialize :class:`TimeAwareObservationV0`.

        Args:
            env: The environment to apply the wrapper
            flatten: Flatten the observation to a `Box` of a single dimension
            normalize_time: if `True` return time in the range [0,1]
                otherwise return time as remaining timesteps before truncation
        """
        super().__init__(env)
        self.flatten = flatten
        self.normalize_time = normalize_time
        self.max_timesteps = getattr(env, "_max_episode_steps")

        if self.normalize_time:
            self._get_time_observation = lambda: self.timesteps / self.max_timesteps
            time_space = Box(0, 1)
        else:
            self._get_time_observation = lambda: self.max_timesteps - self.timesteps
            time_space = Box(0, self.max_timesteps)

        self.time_aware_observation_space = Dict(
            obs=env.observation_space, time=time_space
        )

        if self.flatten:
            self.observation_space = spaces.flatten_space(
                self.time_aware_observation_space
            )
            self._observation_postprocess = lambda observation: spaces.flatten(
                self.time_aware_observation_space, observation
            )
        else:
            self.observation_space = self.time_aware_observation_space
            self._observation_postprocess = lambda observation: observation

    def observation(self, observation: ObsType):
        """Adds to the observation with the current time information.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time information appended to
        """
        time_observation = self._get_time_observation()
        observation = OrderedDict(obs=observation, time=time_observation)

        return self._observation_postprocess(observation)

    def step(self, action: ActType):
        """Steps through the environment, incrementing the time step.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        self.timesteps += 1
        observation, reward, terminated, truncated, info = super().step(action)

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment setting the time to zero.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        self.timesteps = 0
        return super().reset(**kwargs)
