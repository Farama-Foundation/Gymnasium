"""A synchronous vector environment implementation, equivalent to for loop through a number of environments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from gymnasium import Space
from gymnasium.core import ActType, Env, ObsType, RenderFrame
from gymnasium.experimental.vector import VectorEnv
from gymnasium.experimental.vector.vector_env import VectorActType, VectorObsType
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate


__all__ = ["SyncVectorEnv"]


class SyncVectorEnv(VectorEnv[VectorObsType, VectorActType, np.ndarray]):
    """Vectorized environment that serially runs multiple environments.

    Example::

        >>> import gymnasium as gym
        >>> env = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> env.reset()  # doctest: +SKIP
        array([[-0.8286432 ,  0.5597771 ,  0.90249056],
               [-0.85009176,  0.5266346 ,  0.60007906]], dtype=float32)
    """

    def __init__(
        self,
        envs: Iterable[Callable[[], Env[ObsType, ActType]]]
        | Sequence[Env[ObsType, ActType]],
        copy: bool = True,
        render_mode: str | None = None,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            envs: A sequence of environments or functions that generate environments
            copy: Copy the observation on `reset` and `step` functions
            render_mode: The render_mode of the environment
        """
        envs = tuple(envs)
        if all(callable(env_fn) for env_fn in envs):
            envs = [env_fn() for env_fn in envs]
        assert all(isinstance(env, Env) for env in envs)
        self.envs: Sequence[Env[ObsType, ActType]] = envs
        self.num_envs = len(self.envs)

        self.metadata = self.envs[0].metadata
        self.render_mode = render_mode
        self.copy = copy

        assert len(envs) > 0
        self.single_observation_space: Space[ObsType] = self.envs[0].observation_space
        assert all(
            env.observation_space == self.single_observation_space for env in self.envs
        )
        self.single_action_space: Space[ActType] = self.envs[0].action_space
        assert all(env.action_space == self.single_action_space for env in self.envs)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        assert render_mode != "human"
        if render_mode is not None and render_mode.startswith("single_"):
            assert self.envs[0].render_mode == render_mode[len("single_") :] and all(
                env.render_mode is None for env in self.envs[1:]
            )
        else:
            assert all(env.render_mode == render_mode for env in self.envs)

        self._obs_array = create_empty_array(
            self.single_observation_space, self.num_envs
        )
        self._reset_options: dict[str, Any] | None = None
        self._to_reset_envs: np.ndarray = np.full(self.num_envs, dtype=bool)

    def reset(
        self,
        *,
        seed: int | Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VectorObsType, dict[str, Any]]:
        """Resets all the sub-environments.

        Args:
            seed: Resets the environments with a set seed, if an integer is provided, sub-environments are reset with
               seeds [seed + 0, seed + 1, ..., seed + i, ..., seed + self.num_envs - 1]
            options: The options used to reset all the sub-environments with. This option is saved and used on each
               autoreset.

        Returns:
            The reset observation and info of the environment
        """
        self._reset_options = options

        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert isinstance(seed, Sequence) and len(seed) == self.num_envs
        assert all(isinstance(sub_seed, int) or sub_seed is None for sub_seed in seed)

        obs, info = [None for _ in range(self.num_envs)], {}
        for env_num, (env, env_seed) in enumerate(zip(self.envs, seed)):
            env_obs, env_info = env.reset(seed=env_seed, options=self._reset_options)

            obs[env_num] = env_obs
            info = self.add_dict_info(info, env_info, env_num)

        obs = concatenate(self.single_observation_space, obs, self._obs_array)
        if self.copy:
            obs = deepcopy(obs)
        return obs, info

    def step(
        self, actions: VectorActType
    ) -> tuple[VectorObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        env_actions = iterate(self.action_space, actions)

        obs = [None for _ in range(self.num_envs)]
        rewards = [0 for _ in range(self.num_envs)]
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        info = {}

        for env_num, (env, action) in enumerate(zip(self.envs, env_actions)):
            if self._to_reset_envs[env_num]:
                env_obs, env_info = env.reset(options=self._reset_options)

                rewards[env_num] = 0
                terminations[env_num], truncations[env_num] = False, False
            else:
                (
                    env_obs,
                    rewards[env_num],
                    terminations[env_num],
                    truncations[env_num],
                    env_info,
                ) = env.step(action)

            info = self.add_dict_info(info, env_info, env_num)

        obs = concatenate(self.single_observation_space, obs, self._obs_array)
        if self.copy:
            obs = deepcopy(self.copy)
        assert all(reward is not None for reward in rewards)
        rewards = np.array(rewards)
        self._to_reset_envs = np.logical_or(terminations, truncations)

        return obs, rewards, terminations, truncations, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Renders the sub-environments."""
        assert self.render_mode is not None
        if self.render_mode.startswith("single_"):
            return self.envs[0].render()
        else:
            return [env.render() for env in self.envs]

    def close(self):
        """Closes each sub-environments."""
        for env in self.envs:
            env.close()
        super().close()

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Calls a function in each sub-environment with name using args and kwargs return a tuple of results."""
        results = []
        for i, env in enumerate(self.envs):
            try:
                result = getattr(env, name)
            except AttributeError as e:
                raise AttributeError(
                    f"Environment {i} is missing attribute {name}. Full error: {e}"
                )

            if callable(result):
                results.append(result(*args, **kwargs))
            else:
                results.append(result)

        return tuple(results)

    def get_attr(self, name: str) -> tuple[Any, ...]:
        """Gets the attribute from each sub-environment."""
        results = []
        for i, env in enumerate(self.envs):
            try:
                results.append(getattr(env, name))
            except AttributeError as e:
                raise AttributeError(
                    f"Environment {i} is missing attribute {name}. Full error: {e}"
                )

        return tuple(results)

    def set_attr(self, name: str, values: list[Any] | tuple[Any] | Any):
        """Sets an attribute of each sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)
