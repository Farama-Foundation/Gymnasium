"""Wrapper which adds a probability of repeating the previous executed action."""
from typing import Union

import gymnasium as gym
from gymnasium.core import ActType
from gymnasium.error import InvalidProbability


class StickyActionV0(gym.ActionWrapper):
    """Wrapper which adds a probability of repeating the previous action."""

    def __init__(self, env: gym.Env, repeat_action_probability: Union[int, float]):
        """Initialize StickyAction wrapper.

        Args:
            env (Env): the wrapped environment
            repeat_action_probability (int | float): a proability of repeating the old action.
        """
        if not 0 <= repeat_action_probability < 1:
            raise InvalidProbability(
                f"repeat_action_probability should be in the interval [0,1). Received {repeat_action_probability}"
            )
        super().__init__(env)
        self.repeat_action_probability = repeat_action_probability
        self.old_action = None

    def action(self, action: ActType):
        """Execute the action."""
        if (
            self.old_action is not None
            and self.np_random.uniform() < self.repeat_action_probability
        ):
            action = self.old_action
        self.old_action = action
        return action

    def reset(self, **kwargs):
        """Reset the environment."""
        self.old_action = None
        return super().reset(**kwargs)
