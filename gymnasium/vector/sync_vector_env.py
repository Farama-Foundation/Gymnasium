"""A synchronous vector environment."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterator

import numpy as np

from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.utils.spaces import batch_space
from gymnasium.vector.vector_env import (
    VectorActType,
    VectorArrayType,
    VectorEnv,
    VectorObsType,
)


class SyncVectorEnv(VectorEnv[VectorObsType, VectorActType, VectorArrayType]):
    """Vectorized environment that serially runs multiple environments.

    Example::

        >>> import gymnasium as gym
        >>> env = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v0", g=9.81),
        ...     lambda: gym.make("Pendulum-v0", g=1.62)
        ... ])
        >>> env.reset()
        array([[-0.8286432 ,  0.5597771 ,  0.90249056],
               [-0.85009176,  0.5266346 ,  0.60007906]], dtype=float32)
    """

    def __init__(
        self,
        env_fns: Iterator[Callable[[], Env[ObsType, ActType]]],
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        # Add check that envs are the same environment, we should ignore additional wrappers or hyperparameters
        self._check_spaces()

        self.num_envs = len(self.envs)
        self.metadata = self.envs[0].metadata
        self.spec = self.envs[0].spec

        self.single_observation_space: ObsType = self.envs[0].observation_space
        self.single_action_space: ActType = self.envs[0].action_space

        self.observation_space: VectorObsType = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space: VectorActType = batch_space(
            self.single_action_space, self.num_envs
        )

        self._observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

        self.copy = copy

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VectorObsType, dict[str, Any]]:
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation and info of the environment
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminations[:] = False
        self._truncations[:] = False
        observations = []
        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):

            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            observation, info = env.reset(**kwargs)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self._observations = concatenate(
            self.single_observation_space, observations, self._observations
        )
        return (
            deepcopy(self._observations) if self.copy else self._observations
        ), infos

    def step(
        self, actions: VectorActType
    ) -> tuple[
        VectorObsType, VectorArrayType, VectorArrayType, VectorArrayType, dict[str, Any]
    ]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        actions = iterate(self.action_space, actions)

        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            (
                observation,
                self._rewards[i],
                self._terminations[i],
                self._truncations[i],
                info,
            ) = env.step(action)

            if self._terminations[i] or self._truncations[i]:
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self._observations = concatenate(
            self.single_observation_space, observations, self._observations
        )

        return (
            deepcopy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )

    def call(self, name: str, *args: list[Any], **kwargs: Any) -> tuple[Any, ...]:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def get_attr(self, name: str) -> Any:
        """Get a property from each parallel environment.

        Args:
            name (str): Name of the property to be get from each individual environment.

        Returns:
            The property with name
        """
        return self.call(name)

    def set_attr(self, name: str, values: list[Any] | tuple[Any, ...] | Any):
        """Sets an attribute of the sub-environments.

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

    def close(self, **kwargs: Any):
        """Close the environments."""
        for env in self.envs:
            env.close()

    def _check_spaces(self) -> bool:
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        return True
