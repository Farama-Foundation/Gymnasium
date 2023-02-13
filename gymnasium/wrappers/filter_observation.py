"""A wrapper for filtering dictionary observations by their keys."""
import copy
from typing import Sequence

import gymnasium as gym
from gymnasium import spaces


class FilterObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Filter Dict observation space by the keys.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformObservation
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformObservation(env, lambda obs: {'obs': obs, 'time': 0})
        >>> env.observation_space = gym.spaces.Dict(obs=env.observation_space, time=gym.spaces.Discrete(1))
        >>> env.reset(seed=42)
        ({'obs': array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), 'time': 0}, {})
        >>> env = FilterObservation(env, filter_keys=['obs'])
        >>> env.reset(seed=42)
        ({'obs': array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32)}, {})
        >>> env.step(0)
        ({'obs': array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476], dtype=float32)}, 1.0, False, False, {})
    """

    def __init__(self, env: gym.Env, filter_keys: Sequence[str] = None):
        """A wrapper that filters dictionary observations by their keys.

        Args:
            env: The environment to apply the wrapper
            filter_keys: List of keys to be included in the observations. If ``None``, observations will not be filtered and this wrapper has no effect

        Raises:
            ValueError: If the environment's observation space is not :class:`spaces.Dict`
            ValueError: If any of the `filter_keys` are not included in the original `env`'s observation space
        """
        gym.utils.RecordConstructorArgs.__init__(self, filter_keys=filter_keys)
        gym.ObservationWrapper.__init__(self, env)

        wrapped_observation_space = env.observation_space
        if not isinstance(wrapped_observation_space, spaces.Dict):
            raise ValueError(
                f"FilterObservationWrapper is only usable with dict observations, "
                f"environment observation space is {type(wrapped_observation_space)}"
            )

        observation_keys = wrapped_observation_space.spaces.keys()
        if filter_keys is None:
            filter_keys = tuple(observation_keys)

        missing_keys = {key for key in filter_keys if key not in observation_keys}
        if missing_keys:
            raise ValueError(
                "All the filter_keys must be included in the original observation space.\n"
                f"Filter keys: {filter_keys}\n"
                f"Observation keys: {observation_keys}\n"
                f"Missing keys: {missing_keys}"
            )

        self.observation_space = type(wrapped_observation_space)(
            [
                (name, copy.deepcopy(space))
                for name, space in wrapped_observation_space.spaces.items()
                if name in filter_keys
            ]
        )

        self._env = env
        self._filter_keys = tuple(filter_keys)

    def observation(self, observation):
        """Filters the observations.

        Args:
            observation: The observation to filter

        Returns:
            The filtered observations
        """
        filter_observation = self._filter_observation(observation)
        return filter_observation

    def _filter_observation(self, observation):
        observation = type(observation)(
            [
                (name, value)
                for name, value in observation.items()
                if name in self._filter_keys
            ]
        )
        return observation
