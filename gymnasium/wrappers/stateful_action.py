"""``StickyAction`` wrapper - There is a probability that the action is taken again."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.error import InvalidProbability


__all__ = ["StickyAction"]


class StickyAction(
    gym.ActionWrapper[ObsType, ActType, ActType], gym.utils.RecordConstructorArgs
):
    """Adds a probability that the action is repeated for the same ``step`` function.

    This wrapper follows the implementation proposed by `Machado et al., 2018 <https://arxiv.org/pdf/1709.06009.pdf>`_
    in Section 5.2 on page 12.

    No vector version of the wrapper exists.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> env = StickyAction(env, repeat_action_probability=0.9)
        >>> env.reset(seed=123)
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})
        >>> env.step(1)
        (array([ 0.01734283,  0.15089367, -0.02859527, -0.33293587], dtype=float32), 1.0, False, False, {})
        >>> env.step(0)
        (array([ 0.0203607 ,  0.34641072, -0.03525399, -0.6344974 ], dtype=float32), 1.0, False, False, {})
        >>> env.step(1)
        (array([ 0.02728892,  0.5420062 , -0.04794393, -0.9380709 ], dtype=float32), 1.0, False, False, {})
        >>> env.step(0)
        (array([ 0.03812904,  0.34756234, -0.06670535, -0.6608303 ], dtype=float32), 1.0, False, False, {})

    Change logs:
     * v1.0.0 - Initially added
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
