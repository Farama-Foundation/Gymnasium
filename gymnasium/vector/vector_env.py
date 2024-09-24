"""Base class for vectorized environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.utils import seeding


if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec

ArrayType = TypeVar("ArrayType")


__all__ = [
    "VectorEnv",
    "VectorWrapper",
    "VectorObservationWrapper",
    "VectorActionWrapper",
    "VectorRewardWrapper",
    "ArrayType",
]


class VectorEnv(Generic[ObsType, ActType, ArrayType]):
    """Base class for vectorized environments to run multiple independent copies of the same environment in parallel.

    Vector environments can provide a linear speed-up in the steps taken per second through sampling multiple
    sub-environments at the same time. Gymnasium contains two generalised Vector environments: :class:`AsyncVectorEnv`
    and :class:`SyncVectorEnv` along with several custom vector environment implementations.
    For :func:`reset` and :func:`step` batches `observations`, `rewards`,  `terminations`, `truncations` and
    `info` for each sub-environment, see the example below. For the `rewards`, `terminations`, and `truncations`,
    the data is packaged into a NumPy array of shape `(num_envs,)`. For `observations` (and `actions`, the batching
    process is dependent on the type of observation (and action) space, and generally optimised for neural network
    input/outputs. For `info`, the data is kept as a dictionary such that a key will give the data for all sub-environment.

    For creating environments, :func:`make_vec` is a vector environment equivalent to :func:`make` for easily creating
    vector environments that contains several unique arguments for modifying environment qualities, number of environment,
    vectorizer type, vectorizer arguments.

    Note:
        The info parameter of :meth:`reset` and :meth:`step` was originally implemented before v0.25 as a list
        of dictionary for each sub-environment. However, this was modified in v0.25+ to be a dictionary with a NumPy
        array for each key. To use the old info style, utilise the :class:`DictInfoToList` wrapper.

    Examples:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync", wrappers=(gym.wrappers.TimeAwareObservation,))
        >>> envs = gym.wrappers.vector.ClipReward(envs, min_reward=0.2, max_reward=0.8)
        >>> envs
        <ClipReward, SyncVectorEnv(CartPole-v1, num_envs=3)>
        >>> envs.num_envs
        3
        >>> envs.action_space
        MultiDiscrete([2 2 2])
        >>> envs.observation_space
        Box([[-4.80000019        -inf -0.41887903        -inf  0.        ]
         [-4.80000019        -inf -0.41887903        -inf  0.        ]
         [-4.80000019        -inf -0.41887903        -inf  0.        ]], [[4.80000019e+00            inf 4.18879032e-01            inf
          5.00000000e+02]
         [4.80000019e+00            inf 4.18879032e-01            inf
          5.00000000e+02]
         [4.80000019e+00            inf 4.18879032e-01            inf
          5.00000000e+02]], (3, 5), float64)
        >>> observations, infos = envs.reset(seed=123)
        >>> observations
        array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282,  0.        ],
               [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598,  0.        ],
               [ 0.03517495, -0.000635  , -0.01098382, -0.03203924,  0.        ]])
        >>> infos
        {}
        >>> _ = envs.action_space.seed(123)
        >>> actions = envs.action_space.sample()
        >>> observations, rewards, terminations, truncations, infos = envs.step(actions)
        >>> observations
        array([[ 0.01734283,  0.15089367, -0.02859527, -0.33293587,  1.        ],
               [ 0.02909703, -0.16717631,  0.04740972,  0.3319138 ,  1.        ],
               [ 0.03516225, -0.19559774, -0.01162461,  0.25715804,  1.        ]])
        >>> rewards
        array([0.8, 0.8, 0.8])
        >>> terminations
        array([False, False, False])
        >>> truncations
        array([False, False, False])
        >>> infos
        {}
        >>> envs.close()

    To avoid having to wait for all sub-environments to terminated before resetting, implementations will autoreset
    sub-environments on episode end (`terminated or truncated is True`). As a result, when adding observations
    to a replay buffer, this requires knowing when an observation (and info) for each sub-environment are the first
    observation from an autoreset. We recommend using an additional variable to store this information such as
    ``has_autoreset = np.logical_or(terminated, truncated)``.

    The Vector Environments have the additional attributes for users to understand the implementation

    - :attr:`num_envs` - The number of sub-environment in the vector environment
    - :attr:`observation_space` - The batched observation space of the vector environment
    - :attr:`single_observation_space` - The observation space of a single sub-environment
    - :attr:`action_space` - The batched action space of the vector environment
    - :attr:`single_action_space` - The action space of a single sub-environment
    """

    metadata: dict[str, Any] = {}
    spec: EnvSpec | None = None
    render_mode: str | None = None
    closed: bool = False

    observation_space: gym.Space
    action_space: gym.Space
    single_observation_space: gym.Space
    single_action_space: gym.Space

    num_envs: int

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """Reset all parallel environments and return a batch of initial observations and info.

        Args:
            seed: The environment reset seed
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.

        Example:
            >>> import gymnasium as gym
            >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
            >>> observations, infos = envs.reset(seed=42)
            >>> observations
            array([[ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ],
                   [ 0.01522993, -0.04562247, -0.04799704,  0.03392126],
                   [-0.03774345, -0.02418869, -0.00942293,  0.0469184 ]],
                  dtype=float32)
            >>> infos
            {}
        """
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Take an action for each parallel environment.

        Args:
            actions: Batch of actions with the :attr:`action_space` shape.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)

        Note:
            As the vector environments autoreset for a terminating and truncating sub-environments, this will occur on
            the next step after `terminated or truncated is True`.

        Example:
            >>> import gymnasium as gym
            >>> import numpy as np
            >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
            >>> _ = envs.reset(seed=42)
            >>> actions = np.array([1, 0, 1], dtype=np.int32)
            >>> observations, rewards, terminations, truncations, infos = envs.step(actions)
            >>> observations
            array([[ 0.02727336,  0.18847767,  0.03625453, -0.26141977],
                   [ 0.01431748, -0.24002443, -0.04731862,  0.3110827 ],
                   [-0.03822722,  0.1710671 , -0.00848456, -0.2487226 ]],
                  dtype=float32)
            >>> rewards
            array([1., 1., 1.])
            >>> terminations
            array([False, False, False])
            >>> terminations
            array([False, False, False])
            >>> infos
            {}
        """
        raise NotImplementedError(f"{self.__str__()} step function is not implemented.")

    def render(self) -> tuple[RenderFrame, ...] | None:
        """Returns the rendered frames from the parallel environments.

        Returns:
            A tuple of rendered frames from the parallel environments
        """
        raise NotImplementedError(
            f"{self.__str__()} render function is not implemented."
        )

    def close(self, **kwargs: Any):
        """Close all parallel environments and release resources.

        It also closes all the existing image viewers, then calls :meth:`close_extras` and set
        :attr:`closed` as ``True``.

        Warnings:
            This function itself does not close the environments, it should be handled
            in :meth:`close_extras`. This is generic for both synchronous and asynchronous
            vectorized environments.

        Note:
            This will be automatically called when garbage collected or program exited.

        Args:
            **kwargs: Keyword arguments passed to :meth:`close_extras`
        """
        if self.closed:
            return

        self.close_extras(**kwargs)
        self.closed = True

    def close_extras(self, **kwargs: Any):
        """Clean up the extra resources e.g. beyond what's in this base class."""
        pass

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value
        self._np_random_seed = -1

    @property
    def np_random_seed(self) -> int | None:
        """Returns the environment's internal :attr:`_np_random_seed` that if not set will first initialise with a random int as seed.

        If :attr:`np_random_seed` was set directly instead of through :meth:`reset` or :meth:`set_np_random_through_seed`,
        the seed will take the value -1.

        Returns:
            int: the seed of the current `np_random` or -1, if the seed of the rng is unknown
        """
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random_seed

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self

    def _add_info(
        self, vector_infos: dict[str, Any], env_info: dict[str, Any], env_num: int
    ) -> dict[str, Any]:
        """Add env info to the info dictionary of the vectorized environment.

        Given the `info` of a single environment add it to the `infos` dictionary
        which represents all the infos of the vectorized environment.
        Every `key` of `info` is paired with a boolean mask `_key` representing
        whether or not the i-indexed environment has this `info`.

        Args:
            vector_infos (dict): the infos of the vectorized environment
            env_info (dict): the info coming from the single environment
            env_num (int): the index of the single environment

        Returns:
            infos (dict): the (updated) infos of the vectorized environment
        """
        for key, value in env_info.items():
            # If value is a dictionary, then we apply the `_add_info` recursively.
            if isinstance(value, dict):
                array = self._add_info(vector_infos.get(key, {}), value, env_num)
            # Otherwise, we are a base case to group the data
            else:
                # If the key doesn't exist in the vector infos, then we can create an array of that batch type
                if key not in vector_infos:
                    if type(value) in [int, float, bool] or issubclass(
                        type(value), np.number
                    ):
                        array = np.zeros(self.num_envs, dtype=type(value))
                    elif isinstance(value, np.ndarray):
                        # We assume that all instances of the np.array info are of the same shape
                        array = np.zeros(
                            (self.num_envs, *value.shape), dtype=value.dtype
                        )
                    else:
                        # For unknown objects, we use a Numpy object array
                        array = np.full(self.num_envs, fill_value=None, dtype=object)
                # Otherwise, just use the array that already exists
                else:
                    array = vector_infos[key]

                # Assign the data in the `env_num` position
                #   We only want to run this for the base-case data (not recursive data forcing the ugly function structure)
                array[env_num] = value

            # Get the array mask and if it doesn't already exist then create a zero bool array
            array_mask = vector_infos.get(
                f"_{key}", np.zeros(self.num_envs, dtype=np.bool_)
            )
            array_mask[env_num] = True

            # Update the vector info with the updated data and mask information
            vector_infos[key], vector_infos[f"_{key}"] = array, array_mask

        return vector_infos

    def __del__(self):
        """Closes the vector environment."""
        if not getattr(self, "closed", True):
            self.close()

    def __repr__(self) -> str:
        """Returns a string representation of the vector environment.

        Returns:
            A string containing the class name, number of environments and environment spec id
        """
        if self.spec is None:
            return f"{self.__class__.__name__}(num_envs={self.num_envs})"
        else:
            return (
                f"{self.__class__.__name__}({self.spec.id}, num_envs={self.num_envs})"
            )


class VectorWrapper(VectorEnv):
    """Wraps the vectorized environment to allow a modular transformation.

    This class is the base class for all wrappers for vectorized environments. The subclass
    could override some methods to change the behavior of the original vectorized environment
    without touching the original code.

    Note:
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """

    def __init__(self, env: VectorEnv):
        """Initialize the vectorized environment wrapper.

        Args:
            env: The environment to wrap
        """
        self.env = env
        assert isinstance(env, VectorEnv)

        self._observation_space: gym.Space | None = None
        self._action_space: gym.Space | None = None
        self._single_observation_space: gym.Space | None = None
        self._single_action_space: gym.Space | None = None
        self._metadata: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset all environment using seed and options."""
        return self.env.reset(seed=seed, options=options)

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Step through all environments using the actions returning the batched data."""
        return self.env.step(actions)

    def render(self) -> tuple[RenderFrame, ...] | None:
        """Returns the render mode from the base vector environment."""
        return self.env.render()

    def close(self, **kwargs: Any):
        """Close all environments."""
        return self.env.close(**kwargs)

    def close_extras(self, **kwargs: Any):
        """Close all extra resources."""
        return self.env.close_extras(**kwargs)

    @property
    def unwrapped(self):
        """Return the base non-wrapped environment."""
        return self.env.unwrapped

    def __repr__(self):
        """Return the string representation of the vectorized environment."""
        return f"<{self.__class__.__name__}, {self.env}>"

    @property
    def observation_space(self) -> gym.Space:
        """Gets the observation space of the vector environment."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: gym.Space):
        """Sets the observation space of the vector environment."""
        self._observation_space = space

    @property
    def action_space(self) -> gym.Space:
        """Gets the action space of the vector environment."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: gym.Space):
        """Sets the action space of the vector environment."""
        self._action_space = space

    @property
    def single_observation_space(self) -> gym.Space:
        """Gets the single observation space of the vector environment."""
        if self._single_observation_space is None:
            return self.env.single_observation_space
        return self._single_observation_space

    @single_observation_space.setter
    def single_observation_space(self, space: gym.Space):
        """Sets the single observation space of the vector environment."""
        self._single_observation_space = space

    @property
    def single_action_space(self) -> gym.Space:
        """Gets the single action space of the vector environment."""
        if self._single_action_space is None:
            return self.env.single_action_space
        return self._single_action_space

    @single_action_space.setter
    def single_action_space(self, space):
        """Sets the single action space of the vector environment."""
        self._single_action_space = space

    @property
    def num_envs(self) -> int:
        """Gets the wrapped vector environment's num of the sub-environments."""
        return self.env.num_envs

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        return self.env.np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self.env.np_random = value

    @property
    def np_random_seed(self) -> int | None:
        """The seeds of the vector environment's internal :attr:`_np_random`."""
        return self.env.np_random_seed

    @property
    def metadata(self):
        """The metadata of the vector environment."""
        if self._metadata is not None:
            return self._metadata
        return self.env.metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def spec(self) -> EnvSpec | None:
        """Gets the specification of the wrapped environment."""
        return self.env.spec

    @property
    def render_mode(self) -> tuple[RenderFrame, ...] | None:
        """Returns the `render_mode` from the base environment."""
        return self.env.render_mode

    @property
    def closed(self):
        """If the environment has closes."""
        return self.env.closed

    @closed.setter
    def closed(self, value: bool):
        self.env.closed = value


class VectorObservationWrapper(VectorWrapper):
    """Wraps the vectorized environment to allow a modular transformation of the observation.

    Equivalent to :class:`gymnasium.ObservationWrapper` for vectorized environments.
    """

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Modifies the observation returned from the environment ``reset`` using the :meth:`observation`."""
        observations, infos = self.env.reset(seed=seed, options=options)
        return self.observations(observations), infos

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Modifies the observation returned from the environment ``step`` using the :meth:`observation`."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return (
            self.observations(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def observations(self, observations: ObsType) -> ObsType:
        """Defines the vector observation transformation.

        Args:
            observations: A vector observation from the environment

        Returns:
            the transformed observation
        """
        raise NotImplementedError


class VectorActionWrapper(VectorWrapper):
    """Wraps the vectorized environment to allow a modular transformation of the actions.

    Equivalent of :class:`gymnasium.ActionWrapper` for vectorized environments.
    """

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through the environment using a modified action by :meth:`action`."""
        return self.env.step(self.actions(actions))

    def actions(self, actions: ActType) -> ActType:
        """Transform the actions before sending them to the environment.

        Args:
            actions (ActType): the actions to transform

        Returns:
            ActType: the transformed actions
        """
        raise NotImplementedError


class VectorRewardWrapper(VectorWrapper):
    """Wraps the vectorized environment to allow a modular transformation of the reward.

    Equivalent of :class:`gymnasium.RewardWrapper` for vectorized environments.
    """

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through the environment returning a reward modified by :meth:`reward`."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return observations, self.rewards(rewards), terminations, truncations, infos

    def rewards(self, rewards: ArrayType) -> ArrayType:
        """Transform the reward before returning it.

        Args:
            rewards (array): the reward to transform

        Returns:
            array: the transformed reward
        """
        raise NotImplementedError
