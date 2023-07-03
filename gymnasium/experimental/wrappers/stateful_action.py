"""``StickyAction`` wrapper - There is a probability that the action is taken again."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.error import InvalidProbability


__all__ = ["StickyActionV0"]


class StickyActionV0(
    gym.ActionWrapper[ObsType, ActType, ActType], gym.utils.RecordConstructorArgs
):
    """Wrapper which adds a probability of repeating the previous action.

    This wrapper follows the implementation proposed by `Machado et al., 2018 <https://arxiv.org/pdf/1709.06009.pdf>`_
    in Section 5.2 on page 12.
    """

    def __init__(
        self, env: gym.Env[ObsType, ActType], repeat_action_probability: float
    ):
        """Initialize StickyAction wrapper.

        Args:
            env (Env): the wrapped environment
            repeat_action_probability (int | float): a probability of repeating the old action.
        """
        if not 0 <= repeat_action_probability < 1:
            raise InvalidProbability(
                f"repeat_action_probability should be in the interval [0,1). Received {repeat_action_probability}"
            )

        gym.utils.RecordConstructorArgs.__init__(
            self, repeat_action_probability=repeat_action_probability
        )
        gym.ActionWrapper.__init__(self, env)

        self.repeat_action_probability = repeat_action_probability
        self.last_action: ActType | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment."""
        self.last_action = None

        return super().reset(seed=seed, options=options)

    def action(self, action: ActType) -> ActType:
        """Execute the action."""
        if (
            self.last_action is not None
            and self.np_random.uniform() < self.repeat_action_probability
        ):
            action = self.last_action

        self.last_action = action
        return action
