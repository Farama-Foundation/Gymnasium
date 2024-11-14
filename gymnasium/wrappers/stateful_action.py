"""``StickyAction`` wrapper - There is a probability that the action is taken again."""

from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.error import InvalidBound, InvalidProbability


__all__ = ["StickyAction"]


class StickyAction(
    gym.ActionWrapper[ObsType, ActType, ActType], gym.utils.RecordConstructorArgs
):
    """Adds a probability that the action is repeated for the same ``step`` function.

    This wrapper follows the implementation proposed by `Machado et al., 2018 <https://arxiv.org/pdf/1709.06009.pdf>`_
    in Section 5.2 on page 12, and adds the possibility to repeat the action for
    more than one step.

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
     * v1.1.0 - Add `repeat_action_duration` argument for dynamic number of sticky actions
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        repeat_action_probability: float,
        repeat_action_duration: int | tuple[int, int] = 1,
    ):
        """Initialize StickyAction wrapper.

        Args:
            env (Env): the wrapped environment,
            repeat_action_probability (int | float): a probability of repeating the old action,
            repeat_action_duration (int | tuple[int, int]): the number of steps
                the action is repeated. It can be either an int (for deterministic
                repeats) or a tuple[int, int] for a range of stochastic number of repeats.
        """
        if not 0 <= repeat_action_probability < 1:
            raise InvalidProbability(
                f"`repeat_action_probability` should be in the interval [0,1). Received {repeat_action_probability}"
            )

        if isinstance(repeat_action_duration, int):
            repeat_action_duration = (repeat_action_duration, repeat_action_duration)

        if not isinstance(repeat_action_duration, tuple):
            raise ValueError(
                f"`repeat_action_duration` should be either an integer or a tuple. Received {repeat_action_duration}"
            )
        elif len(repeat_action_duration) != 2:
            raise ValueError(
                f"`repeat_action_duration` should be a tuple or a list of two integers. Received {repeat_action_duration}"
            )
        elif repeat_action_duration[0] > repeat_action_duration[1]:
            raise InvalidBound(
                f"`repeat_action_duration` is not a valid bound. Received {repeat_action_duration}"
            )
        elif np.any(np.array(repeat_action_duration) < 1):
            raise ValueError(
                f"`repeat_action_duration` should be larger or equal than 1. Received {repeat_action_duration}"
            )

        gym.utils.RecordConstructorArgs.__init__(
            self, repeat_action_probability=repeat_action_probability
        )
        gym.ActionWrapper.__init__(self, env)

        self.repeat_action_probability = repeat_action_probability
        self.repeat_action_duration_range = repeat_action_duration

        self.last_action: ActType | None = None
        self.is_sticky_actions: bool = False  # if sticky actions are taken
        self.num_repeats: int = 0  # number of sticky action repeats
        self.repeats_taken: int = 0  # number of sticky actions taken

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment."""
        self.last_action = None
        self.is_sticky_actions = False
        self.num_repeats = 0
        self.repeats_taken = 0

        return super().reset(seed=seed, options=options)

    def action(self, action: ActType) -> ActType:
        """Execute the action."""
        # either the agent was already "stuck" into repeats, or a new series of repeats is triggered
        if self.is_sticky_actions or (
            self.last_action is not None
            and self.np_random.uniform() < self.repeat_action_probability
        ):
            # if a new series starts, randomly sample its duration
            if self.num_repeats == 0:
                self.num_repeats = self.np_random.integers(
                    self.repeat_action_duration_range[0],
                    self.repeat_action_duration_range[1] + 1,
                )
            action = self.last_action
            self.is_sticky_actions = True
            self.repeats_taken += 1

        # repeats are done, reset "stuck" status
        if self.is_sticky_actions and self.num_repeats == self.repeats_taken:
            self.is_sticky_actions = False
            self.num_repeats = 0
            self.repeats_taken = 0

        self.last_action = action
        return action
