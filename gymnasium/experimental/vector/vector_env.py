"""Base class for vectorized environments."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

import gymnasium as gym
from gymnasium import Space
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.utils import seeding


if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


__all__ = [
    "VectorEnv",
    "VectorWrapper",
    "VectorObservationWrapper",
    "VectorActionWrapper",
    "VectorRewardWrapper",
    "VectorObsType",
    "VectorActType",
    "VectorArrayType",
    "VectorWrapperObsType",
    "VectorWrapperActType",
    "VectorWrapperArrayType",
]

VectorObsType = TypeVar("VectorObsType")
VectorActType = TypeVar("VectorActType")
VectorArrayType = TypeVar("VectorArrayType")


class VectorEnv(Generic[VectorObsType, VectorActType, VectorArrayType]):
    """Base class for vectorized environments to run multiple independent copies of the same environment in parallel.

    Vector environments can provide a linear speed-up in the steps taken per second through sampling multiple
    sub-environments at the same time. To prevent terminated environments waiting until all sub-environments have
    terminated or truncated, the vector environments autoreset sub-environments after they terminate or truncated.
    As a result, the final step's observation and info are overwritten by the reset's observation and info.
    Therefore, the observation and info for the final step of a sub-environment is stored in the info parameter,
    using `"final_observation"` and `"final_info"` respectively. See :meth:`step` for more information.

    The vector environments batch `observations`, `rewards`, `terminations`, `truncations` and `info` for each
    parallel environment. In addition, :meth:`step` expects to receive a batch of actions for each parallel environment.

    Gymnasium contains two types of Vector environments: :class:`AsyncVectorEnv` and :class:`SyncVectorEnv`.

    The Vector Environments have the additional attributes for users to understand the implementation

    - :attr:`num_envs` - The number of sub-environment in the vector environment
    - :attr:`observation_space` - The batched observation space of the vector environment
    - :attr:`single_observation_space` - The observation space of a single sub-environment
    - :attr:`action_space` - The batched action space of the vector environment
    - :attr:`single_action_space` - The action space of a single sub-environment

    Note:
        The info parameter of :meth:`reset` and :meth:`step` was originally implemented before OpenAI Gym v25 was a list
        of dictionary for each sub-environment. However, this was modified in OpenAI Gym v25+ and in Gymnasium to a
        dictionary with a NumPy array for each key. To use the old info style using the :class:`VectorListInfo`.

    Note:
        To render the sub-environments, use :meth:`call` with "render" arguments. Remember to set the `render_modes`
        for all the sub-environments during initialization.

    Note:
        All parallel environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported.
    """

    # Set this in the subclasses
    metadata: dict[str, Any] = {"render_modes": []}
    # Set the render mode if rendering is enabled
    render_mode: str | None = None
    # Set the env spec for the sub-environment or vector environment
    spec: EnvSpec | None = None
    # Closed
    closed: bool = False

    # The obs and action space, set in all subclasses
    observation_space: gym.Space[VectorObsType]
    action_space: gym.Space[VectorActType]
    single_observation_space: gym.Space[ObsType]
    single_action_space: gym.Space[ActType]

    # The number of environments that are vectorised
    num_envs: int

    # The random number generator for the environment (possibly not sub-environments)
    _np_random: np.random.Generator | None = None

    def step(
        self, actions: VectorActType
    ) -> tuple[
        VectorObsType, VectorArrayType, VectorArrayType, VectorArrayType, dict[str, Any]
    ]:
        """Take an action for each parallel environment.

        Args:
            actions: element of :attr:`action_space` Batch of actions.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)

        Note:
            As the vector environments autoreset for a terminating and truncating sub-environments,
            the returned observation and info is not the final step's observation or info which is instead stored in
            info as `"final_observation"` and `"final_info"`.

        Example:
            >>> import gymnasium as gym
            >>> import numpy as np
            >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
            >>> _ = envs.reset(seed=42)
            >>> actions = np.array([1, 0, 1])
            >>> observations, rewards, termination, truncation, infos = envs.step(actions)
            >>> observations
            array([[ 0.02727336,  0.18847767,  0.03625453, -0.26141977],
                   [ 0.01431748, -0.24002443, -0.04731862,  0.3110827 ],
                   [-0.03822722,  0.1710671 , -0.00848456, -0.2487226 ]],
                  dtype=float32)
            >>> rewards
            array([1., 1., 1.])
            >>> termination
            array([False, False, False])
            >>> termination
            array([False, False, False])
            >>> infos
            {}
        """
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VectorObsType, dict[str, Any]]:  # type: ignore
        """Reset all sub-environments and return a batch of initial observations and info.

        Args:
            seed: The environment reset seeds
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Renders the sub-environment depending on the render modes of the sub-environments."""
        raise NotImplementedError

    def close(self):
        """Close all parallel environments and release resources.

        It also closes all the existing image viewers, then calls :meth:`close_extras` and set
        :attr:`closed` as ``True``.

        Warnings:
            This function itself does not close the environments, it should be handled
            in :meth:`close_extras`. This is generic for both synchronous and asynchronous
            vectorized environments.

        Note:
            This will be automatically called when garbage collected or program exited.
        """
        self.closed = True

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def __del__(self):
        """Closes the vector environment."""
        if not getattr(self, "closed", True):
            self.close()

    def __str__(self) -> str:
        """Returns a string representation of the vector environment.

        Returns:
            A string containing the class name, number of environments and environment spec id
        """
        if self.spec is None:
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Support with-statement for the environment and closes the environment."""
        self.close()
        # propagate exception
        return False

    def transform_list_info_to_dict(
        self, infos: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Transforms a list of sub-environment info to an equivalent dictionary based version.

        Args:
            infos: List of sub-environment infos

        Returns:
            A dictionary based version of the list of infos
        """
        vector_info = {}
        for i, info in enumerate(infos):
            vector_info = self.add_dict_info(vector_info, info, i)

        return vector_info

    def add_dict_info(
        self, vector_info: dict[str, Any], subenv_info: dict[str, Any], subenv_num: int
    ):
        """Add env info to the info dictionary of the vectorized environment.

        Given the `info` of a single environment add it to the `infos` dictionary
        which represents all the infos of the vectorized environment.
        Every `key` of `info` is paired with a boolean mask `_key` representing
        whether or not the i-indexed environment has this `info`.

        Args:
            vector_info: the infos of the vectorized environment
            subenv_info: the info coming from the single environment
            subenv_num: the index of the single environment

        Returns:
            updated vector info: the (updated) infos of the vectorized environment
        """
        for subenv_key, subenv_value in subenv_info.items():
            if subenv_key not in vector_info:
                info_array, array_mask = self.init_info_array(type(subenv_value))
            else:
                info_array, array_mask = subenv_value, vector_info[f"_{subenv_key}"]

            info_array[subenv_num], array_mask[subenv_num] = subenv_value, True
            vector_info[subenv_key], vector_info[f"_{subenv_key}"] = (
                info_array,
                array_mask,
            )
        return vector_info

    def init_info_array(self, dtype: type) -> tuple[np.ndarray, np.ndarray]:
        """Initialize the info array.

        Initialize the info array. If the dtype is numeric
        the info array will have the same dtype, otherwise
        will be an array of `None`. Also, a boolean array
        of the same length is returned. It will be used for
        assessing which environment has info data.

        Args:
            dtype (type): data type of the info coming from the env.

        Returns:
            array (np.ndarray): the initialized info array.
            array_mask (np.ndarray): the initialized boolean array.
        """
        if dtype in [int, float, bool] or issubclass(dtype, np.number):
            array = np.zeros(self.num_envs, dtype=dtype)
        else:
            array = np.full(self.num_envs, None, dtype=object)

        array_mask = np.zeros(self.num_envs, dtype=bool)
        return array, array_mask


VectorWrapperObsType = TypeVar("VectorWrapperObsType")
VectorWrapperActType = TypeVar("VectorWrapperActType")
VectorWrapperArrayType = TypeVar("VectorWrapperArrayType")


class VectorWrapper(
    VectorEnv[VectorWrapperObsType, VectorWrapperActType, VectorWrapperArrayType]
):
    """Wraps the vectorized environment to allow a modular transformation.

    This class is the base class for all wrappers for vectorized environments. The subclass
    could override some methods to change the behavior of the original vectorized environment
    without touching the original code.

    Note:
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """

    def __init__(self, env: VectorEnv[VectorObsType, VectorActType, VectorArrayType]):
        """Constructor for the wrapper allowing new observation and action space with metadata."""
        assert isinstance(env, VectorEnv)
        self.env = env

        self._action_space: Space[VectorWrapperObsType] | None = None
        self._observation_space: Space[VectorWrapperActType] | None = None
        self._metadata: dict[str, Any] | None = None

    # explicitly forward the methods defined in VectorEnv
    # to self.env (instead of the base class)

    def step(
        self, actions: VectorActType
    ) -> tuple[
        VectorObsType, VectorArrayType, VectorArrayType, VectorArrayType, dict[str, Any]
    ]:
        """Steps through the environment with actions, returning the observations, rewards, terminations, truncation and info."""
        return self.env.step(actions)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VectorObsType, dict[str, Any]]:
        """Resets the environment with a seed and options, returning the observation and info."""
        return self.env.reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Renders the environment, returning the results."""
        return self.env.render()

    def close(self):
        """Closes the environment."""
        return self.env.close()

    # implicitly forward all other methods and attributes to self.env
    def __getattr__(self, name: str) -> Any:
        """Gets attributes from the environment if the attribute doesn't start with an underscore."""
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    @property
    def spec(self) -> EnvSpec | None:
        """Returns the EnvSpec of the base vector environment."""
        return self.env.spec

    @property
    def unwrapped(self) -> VectorEnv[VectorObsType, VectorActType, VectorArrayType]:
        """Unwrap the environment to the base vector environment."""
        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        """Returns the number of environments in the vector environment."""
        return self.env.num_envs

    @property
    def class_name(self) -> str:
        """Returns the wrapper's name."""
        return self.__name__

    @property
    def action_space(
        self,
    ) -> Space[VectorActType] | Space[VectorWrapperActType]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: Space[VectorWrapperActType]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> Space[VectorObsType] | Space[VectorWrapperObsType]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: Space[VectorWrapperObsType]):
        self._observation_space = space

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the :attr:`Env` :attr:`metadata`."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]):
        self._metadata = value

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the :attr:`Env` :attr:`np_random` attribute."""
        return self.env.np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self.env.np_random = value

    @property
    def _np_random(self):
        """This code will never be run due to __getattr__ being called prior this.

        It seems that @property overwrites the variable (`_np_random`) meaning that __getattr__ gets called with the missing variable.
        """
        raise AttributeError(
            "Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`."
        )

    def __str__(self) -> str:
        """Adds the wrappers name with the wrapper's environment string."""
        return f"<{self.__class__.__name__}, {self.env}>"

    def __del__(self):
        """If the environment is not :attr:`closed` already, close the environments."""
        if not self.closed:
            self.close()


class VectorObservationWrapper(
    VectorWrapper[VectorWrapperObsType, VectorActType, VectorArrayType]
):
    """Wraps the vectorized environment to allow a modular transformation of the observation. Equivalent to :class:`gym.ObservationWrapper` for vectorized environments."""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VectorObsType, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.reset` using the modified ``observations`` from :meth:`self.observations`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, actions: VectorActType
    ) -> tuple[
        VectorObsType, VectorArrayType, VectorArrayType, VectorArrayType, dict[str, Any]
    ]:
        """Runs the :attr:`env` :meth:`env.step` using the modified ``observations`` from :meth:`self.observations`."""
        obs, reward, terminations, truncations, info = self.env.step(actions)
        return self.observations(obs), reward, terminations, truncations, info

    def observations(self, obs: VectorObsType) -> VectorWrapperObsType:
        """Defines the observation transformation.

        Args:
            obs: The observation from the environment

        Returns:
            The transformed observation
        """
        raise NotImplementedError


class VectorActionWrapper(
    VectorWrapper[VectorObsType, VectorWrapperActType, VectorArrayType]
):
    """Wraps the vectorized environment to allow a modular transformation of the actions. Equivalent of :class:`~gym.ActionWrapper` for vectorized environments."""

    def step(
        self, actions: VectorWrapperActType
    ) -> tuple[
        VectorObsType, VectorArrayType, VectorArrayType, VectorArrayType, dict[str, Any]
    ]:
        """Runs the :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        return self.env.step(self.actions(actions))

    def actions(self, actions: VectorWrapperActType) -> VectorActType:
        """Transform the actions before sending them to the environment.

        Args:
            actions: the actions to transform

        Returns:
            The transformed actions
        """
        raise NotImplementedError


class VectorRewardWrapper(VectorWrapper[VectorObsType, VectorActType, VectorArrayType]):
    """Wraps the vectorized environment to allow a modular transformation of the reward. Equivalent of :class:`~gym.RewardWrapper` for vectorized environments."""

    def step(
        self, actions: VectorActType
    ) -> tuple[
        VectorObsType, VectorArrayType, VectorArrayType, VectorArrayType, dict[str, Any]
    ]:
        """Runs the :attr:`env` :meth:`env.step` using the modified ``rewards`` from :meth:`self.rewards`."""
        obs, reward, termination, truncation, info = self.env.step(actions)
        return obs, self.rewards(reward), termination, truncation, info

    def rewards(self, reward: VectorArrayType) -> VectorArrayType:
        """Transform the reward before returning it.

        Args:
            reward (array): the reward to transform

        Returns:
            array: the transformed reward
        """
        raise NotImplementedError
