"""Wrapper for delaying the returned observation."""

from collections import deque

import jumpy as jp

import gymnasium as gym
from gymnasium.core import ObsType


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
        super().__init__(env)
        self.delay = delay
        self.observation_queue = deque()

    def observation(self, observation: ObsType) -> ObsType:
        """Return the delayed observation."""
        self.observation_queue.append(observation)

        if len(self.observation_queue) > self.delay:
            return self.observation_queue.popleft()

        return jp.zeros_like(observation)
