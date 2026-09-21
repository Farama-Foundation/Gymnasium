"""Stateful action wrappers - ``StickyAction``, ``RepeatAction`` and ``ActionChunk``."""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.error import InvalidBound, InvalidProbability
from gymnasium.wrappers.common import Autoreset

__all__ = ["StickyAction", "RepeatAction", "ActionChunk"]


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

        # The pair of bounds that the number of repeats is drawn from.
        # `repeat_action_duration` keeps the value the caller gave, which is what
        # `RecordConstructorArgs` stores for the env spec.
        if isinstance(repeat_action_duration, int):
            duration_range = (repeat_action_duration, repeat_action_duration)
        else:
            duration_range = repeat_action_duration

        if not isinstance(duration_range, tuple):
            raise ValueError(
                f"`repeat_action_duration` should be either an integer or a tuple. Received {duration_range}"
            )
        elif len(duration_range) != 2:
            raise ValueError(
                f"`repeat_action_duration` should be a tuple or a list of two integers. Received {duration_range}"
            )
        elif duration_range[0] > duration_range[1]:
            raise InvalidBound(
                f"`repeat_action_duration` is not a valid bound. Received {duration_range}"
            )
        elif np.any(np.array(duration_range) < 1):
            raise ValueError(
                f"`repeat_action_duration` should be larger or equal than 1. Received {duration_range}"
            )

        gym.utils.RecordConstructorArgs.__init__(
            self,
            repeat_action_probability=repeat_action_probability,
            repeat_action_duration=repeat_action_duration,
        )
        gym.ActionWrapper.__init__(self, env)

        self.repeat_action_probability = repeat_action_probability
        self.repeat_action_duration_range = duration_range

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


class RepeatAction(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Repeats the given action for ``num_repeats`` steps, accumulating the reward.

    This is useful for environments where the agent does not need to make
    a decision at every time step. By repeating actions, the effective
    decision frequency is reduced, which can speed up training and make
    exploration more efficient.

    Unlike :class:`StickyAction`, which *stochastically* replaces the agent's
    action with the *previous* action, ``ActionRepeat`` *deterministically*
    executes the *current* action multiple times and returns the accumulated
    reward together with the final observation. If the episode terminates or
    is truncated during repetition, the loop stops early.

    No vector version of the wrapper exists.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> env = RepeatAction(env, num_repeats=4)
        >>> obs, info = env.reset(seed=123)
        >>> obs, reward, terminated, truncated, info = env.step(1)
        >>> reward  # sum of 4 inner-step rewards
        4.0

    Change logs:
     * v1.3.0 - Initially added
    """

    def __init__(self, env: gym.Env[ObsType, ActType], num_repeats: int):
        """Initialize ActionRepeat wrapper.

        Args:
            env (Env): The wrapped environment.
            num_repeats (int): The number of times to repeat each action.
        """
        if not np.issubdtype(type(num_repeats), np.integer):
            raise TypeError(
                f"The num_repeats is expected to be an integer, actual type: {type(num_repeats)}"
            )
        if num_repeats < 1:
            raise ValueError(
                f"The num_repeats value needs to be equal or greater than one, actual value: {num_repeats}"
            )

        gym.utils.RecordConstructorArgs.__init__(self, num_repeats=num_repeats)
        gym.Wrapper.__init__(self, env)

        self.num_repeats = num_repeats

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment, repeating the action for ``num_repeats`` steps.

        The reward returned is the sum of rewards from all inner steps.
        If the episode terminates or is truncated during repetition,
        the loop stops early.

        Args:
            action: The action to repeat.

        Returns:
            The last observation, accumulated reward, terminated, truncated, and info.
        """
        total_reward = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        for _ in range(self.num_repeats):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ActionChunk(
    gym.Wrapper[ObsType, np.ndarray, ObsType, np.ndarray],
    gym.utils.RecordConstructorArgs,
):
    """Execute a fixed-length sequence of distinct Box actions in one outer step.

    The action space has shape ``(chunk_size, *env.action_space.shape)``. The
    entire NumPy action array is validated before executing any action. Execution
    stops on termination or truncation and returns the last observation, the sum
    of the executed rewards, and the last inner info dictionary. No reset occurs
    inside a chunk. Call :meth:`reset` before the first chunk, after an episode
    ends, or after an inner step fails.

    ``info[info_key]`` contains float64 ``rewards`` and boolean ``valid_mask``
    arrays of length ``chunk_size``, plus ``length`` (the executed step count).
    Unexecuted entries contain zero and False. Other intermediate observations
    and info dictionaries are not retained. The reserved info key and its
    underscore-prefixed vector mask key must not occur in inner reset/step info.

    Place TimeLimit, RecordEpisodeStatistics and FrameStackObservation inside
    this wrapper to count individual environment steps, or outside to count
    chunks. An inner Autoreset or ActionChunk wrapper is rejected. An outer
    Autoreset is supported; its reset-only step discards the whole input chunk.
    Custom environments that reset themselves inside step are unsupported.

    To preserve a per-step discounted objective, training code can compute
    ``sum(gamma**i * rewards[i] for i in range(length))`` and bootstrap with
    ``gamma**length`` (respecting termination). The returned reward is an
    undiscounted scalar sum; a fixed chunk-level discount generally describes a
    different objective when chunks end early.

    No vector wrapper is provided. Wrap each scalar environment before creating
    a SyncVectorEnv or AsyncVectorEnv. Under same-step autoreset, the completed
    chunk's metadata is in ``final_info``; reset-only next-step calls have no
    chunk metadata. Vector info presence masks distinguish these cases.

    Example:
        >>> import gymnasium as gym
        >>> env = ActionChunk(gym.make("Pendulum-v1"), chunk_size=4)
        >>> observation, info = env.reset(seed=42)
        >>> observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        >>> info["action_chunk"]["length"]
        4
        >>> env.close()
    """

    def __init__(
        self,
        env: gym.Env[ObsType, np.ndarray],
        chunk_size: int,
        *,
        info_key: str = "action_chunk",
    ):
        """Initialize sequential action execution for a single Box-action environment.

        Args:
            env: The scalar environment, without an inner autoreset wrapper.
            chunk_size: Positive number of actions supplied per outer step.
            info_key: Nonempty info namespace, which must not start with an underscore.
        """
        if not isinstance(env, gym.Env) or not isinstance(
            env.action_space, gym.spaces.Box
        ):
            raise TypeError(
                "ActionChunk requires a scalar environment with a Box action space."
            )
        if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(
            chunk_size, (int, np.integer)
        ):
            raise TypeError("chunk_size must be an integer.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if not isinstance(info_key, str) or not info_key or info_key.startswith("_"):
            raise ValueError(
                "info_key must be a nonempty string without a leading underscore."
            )
        inner = env
        while isinstance(inner, gym.Wrapper):
            if isinstance(inner, (Autoreset, ActionChunk)):
                raise ValueError(
                    "ActionChunk cannot contain an Autoreset or another ActionChunk wrapper."
                )
            inner = inner.env

        gym.utils.RecordConstructorArgs.__init__(
            self, chunk_size=chunk_size, info_key=info_key
        )
        gym.Wrapper.__init__(self, env)
        self.chunk_size = int(chunk_size)
        self.info_key = info_key
        self.action_space = gym.vector.utils.batch_space(
            env.action_space, self.chunk_size
        )
        self._needs_reset = True

    def _check_info(self, info: dict[str, Any]) -> None:
        """Keep chunk metadata and vector presence masks free of collisions."""
        if self.info_key in info or f"_{self.info_key}" in info:
            raise ValueError(
                f"ActionChunk reserved info key collision: {self.info_key!r}. Choose another info_key."
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and allow a new chunk after a successful reset."""
        self._needs_reset = True
        observation, info = self.env.reset(seed=seed, options=options)
        self._check_info(info)
        self._needs_reset = False
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Validate the complete action block, then execute until it ends or the episode finishes."""
        if self._needs_reset:
            raise gym.error.ResetNeeded(
                "Call reset before executing another action chunk."
            )
        if not isinstance(action, np.ndarray):
            raise TypeError("ActionChunk actions must be NumPy arrays.")
        if action not in self.action_space:
            raise ValueError(f"Action chunk must belong to {self.action_space}.")
        actions = action.copy()
        rewards = np.zeros(self.chunk_size, dtype=np.float64)
        valid_mask = np.zeros(self.chunk_size, dtype=np.bool_)
        total_reward = 0.0
        # Any failure after execution begins leaves the environment requiring reset.
        self._needs_reset = True
        for index in range(self.chunk_size):
            inner_action = np.array(
                actions[index], dtype=self.env.action_space.dtype, copy=True
            )
            observation, reward, terminated, truncated, info = self.env.step(
                inner_action
            )
            self._check_info(info)
            if isinstance(reward, np.ndarray) and reward.ndim != 0:
                raise TypeError(
                    "ActionChunk requires a scalar reward for every inner step."
                )
            reward_value = float(reward)
            rewards[index] = reward_value
            total_reward += reward_value
            valid_mask[index] = True
            if terminated or truncated:
                break
        info = dict(info)
        info[self.info_key] = {
            "rewards": rewards,
            "valid_mask": valid_mask,
            "length": index + 1,
        }
        self._needs_reset = bool(terminated or truncated)
        return observation, total_reward, terminated, truncated, info
