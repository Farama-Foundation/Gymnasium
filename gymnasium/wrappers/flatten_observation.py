"""Wrapper for flattening observations of an environment."""
import gymnasium as gym
from gymnasium import spaces


class FlattenObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that flattens the observation.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FlattenObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (27648,)
    """

    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """
        return spaces.flatten(self.env.observation_space, observation)
