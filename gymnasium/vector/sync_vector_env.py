"""Implementation of a synchronous (for loop) vectorization method of any environment."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from gymnasium import Env
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import ArrayType, VectorEnv


__all__ = ["SyncVectorEnv"]


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("Pendulum-v1", num_envs=2, vectorization_mode="sync")
        >>> envs
        SyncVectorEnv(Pendulum-v1, num_envs=2)
        >>> envs = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> envs
        SyncVectorEnv(num_envs=2)
        >>> obs, infos = envs.reset(seed=42)
        >>> obs
        array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32)
        >>> infos
        {}
        >>> _ = envs.action_space.seed(42)
        >>> actions = envs.action_space.sample()
        >>> obs, rewards, terminates, truncates, infos = envs.step(actions)
        >>> obs
        array([[-0.1878752 ,  0.98219293,  0.7695615 ],
               [ 0.6102389 ,  0.79221743, -0.8498053 ]], dtype=float32)
        >>> rewards
        array([-2.96562607, -0.99902063])
        >>> terminates
        array([False, False])
        >>> truncates
        array([False, False])
        >>> infos
        {}
        >>> envs.close()
    """

    def __init__(
        self,
        env_fns: Iterator[Callable[[], Env]] | Sequence[Callable[[], Env]],
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
        self.copy = copy
        self.env_fns = env_fns

        # Initialise all sub-environments
        self.envs = [env_fn() for env_fn in env_fns]

        # Define core attributes using the sub-environments
        # As we support `make_vec(spec)` then we can't include a `spec = self.envs[0].spec` as this doesn't guarantee we can actual recreate the vector env.
        self.num_envs = len(self.envs)
        self.metadata = self.envs[0].metadata
        self.render_mode = self.envs[0].render_mode

        # Initialises the single spaces from the sub-environments
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self._check_spaces()

        # Initialise the obs and action space based on the single versions and num of sub-environments
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # Initialise attributes used in `step` and `reset`
        self._observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets each of the sub-environments and concatenate the results together.

        Args:
            seed: Seeds used to reset the sub-environments, either
                * ``None`` - random seeds for all environment
                * ``int`` - ``[seed, seed+1, ..., seed+n]``
                * List of ints - ``[1, 2, 3, ..., n]``
            options: Option information used for each sub-environment

        Returns:
            Concatenated observations and info from each sub-environment
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

        observations, infos = [], {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            env_obs, env_info = env.reset(seed=single_seed, options=options)

            observations.append(env_obs)
            infos = self._add_info(infos, env_info, i)

        # Concatenate the observations
        self._observations = concatenate(
            self.single_observation_space, observations, self._observations
        )

        return deepcopy(self._observations) if self.copy else self._observations, infos

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        actions = iterate(self.action_space, actions)

        observations, infos = [], {}
        for i, action in enumerate(actions):
            if self._autoreset_envs[i]:
                env_obs, env_info = self.envs[i].reset()

                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
            else:
                (
                    env_obs,
                    self._rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    env_info,
                ) = self.envs[i].step(action)

            observations.append(env_obs)
            infos = self._add_info(infos, env_info, i)

        # Concatenate the observations
        self._observations = concatenate(
            self.single_observation_space, observations, self._observations
        )
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            deepcopy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )

    def render(self) -> tuple[RenderFrame, ...] | None:
        """Returns the rendered frames from the environments."""
        return tuple(env.render() for env in self.envs)

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Calls a sub-environment method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = env.get_wrapper_attr(name)

            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def get_attr(self, name: str) -> Any:
        """Get a property from each parallel environment.

        Args:
            name (str): Name of the property to get from each individual environment.

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
                "Values must be a list or tuple with length equal to the number of environments. "
                f"Got `{len(values)}` values for {self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            env.set_wrapper_attr(name, value)

    def close_extras(self, **kwargs: Any):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self) -> bool:
        """Check that each of the environments obs and action spaces are equivalent to the single obs and action space."""
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    f"Some environments have an observation space different from `{self.single_observation_space}`. "
                    "In order to batch observations, the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    f"Some environments have an action space different from `{self.single_action_space}`. "
                    "In order to batch actions, the action spaces from all environments must be equal."
                )

        return True
