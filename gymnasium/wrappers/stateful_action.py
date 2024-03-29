"""A collection of wrappers for modifying actions.

* ``StickyAction`` wrapper - There is a probability that the action is taken again.
* ``RepeatAction`` wrapper - Repeat a single action multiple times.
"""
from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.error import InvalidProbability


__all__ = ["StickyAction", "RepeatAction"]


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


class RepeatAction(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Repeatedly executes a given action in the underlying environment.

    Upon calling the `step` method of this wrapper, `num_repeats`-many steps will be taken
    with the same action in the underlying environment.
    The wrapper sums the rewards collected from the underlying environment and returns the last
    environment state observed.
    If a termination or truncation is encountered during these steps, the wrapper will stop prematurely.
    The `info` will additionally contain a field `"num_action_repetitions"`, which specifies
    how many steps were actually taken.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> wrapped = RepeatAction(env, num_repeats=2)
        >>> env.reset(seed=123)
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})
        >>> env.step(0)
        (array([ 0.01734283, -0.23932791, -0.02859527,  0.25216764], dtype=float32), 1.0, False, False, {})
        >>> env.step(0)         # Perform the same action again
        (array([ 0.01255627, -0.43403012, -0.02355192,  0.5356957 ], dtype=float32), 1.0, False, False, {})
        >>> wrapped.reset(seed=123)         # Now we do the same thing with the `RepeatAction` wrapper
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})
        >>> wrapped.step(0)
        (array([ 0.01255627, -0.43403012, -0.02355192,  0.5356957 ], dtype=float32), 2.0, False, False, {'num_action_repetitions': 2})
    """

    def __init__(self, env: gym.Env[ObsType, ActType], num_repeats: int):
        """Initialize RepeatAction wrapper.

        Args:
            env (Env): the wrapped environment
            num_repeats (int): the maximum number of times to repeat the action
        """
        if num_repeats <= 1:
            raise ValueError(
                f"Number of action repeats should be greater than 1, but got {num_repeats}"
            )

        gym.utils.RecordConstructorArgs.__init__(self, num_repeats=num_repeats)
        gym.Wrapper.__init__(self, env)
        self._num_repeats = num_repeats

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Repeat `action` several times.

        This step method will execute `action` at most `num_repeats`-many times in `self.env`,
        or until a termination or truncation is encountered. The reward returned
        is the sum of rewards collected from `self.env`. The last observation from the
        environment is returned.
        """
        num_steps = 0
        total_reward = 0
        assert self._num_repeats > 0
        for _ in range(self._num_repeats):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            num_steps += 1
            if terminated or truncated:
                break
        info["num_action_repetitions"] = num_steps
        return observation, total_reward, terminated, truncated, info
