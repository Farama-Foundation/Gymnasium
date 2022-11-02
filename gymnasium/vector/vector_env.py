"""Base class for vectorized environments."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.utils import seeding

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


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
    - :attr:`action_space` - The batched action space of the vector environment

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

    metadata: dict[str, Any] = {}
    render_mode: str | None = None
    spec: EnvSpec | None = None
    closed: bool = False

    action_space: spaces.Space[VectorActType]
    observation_space: spaces.Space[VectorObsType]
    num_envs: int

    _np_random: np.random.Generator | None = None

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VectorObsType, dict[str, Any]]:  # type: ignore
        """Reset all parallel environments and return a batch of initial observations and info.

        Args:
            seed: The environment reset seeds
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

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
        """
        raise NotImplementedError

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """TODO."""
        raise NotImplementedError

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
        """
        self.closed = True

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self

    def _add_info(
        self, infos: dict[str, Any], info: dict[str, Any], env_num: int
    ) -> dict[str, Any]:
        """Add env info to the info dictionary of the vectorized environment.

        Given the `info` of a single environment add it to the `infos` dictionary
        which represents all the infos of the vectorized environment.
        Every `key` of `info` is paired with a boolean mask `_key` representing
        whether or not the i-indexed environment has this `info`.

        Args:
            infos (dict): the infos of the vectorized environment
            info (dict): the info coming from the single environment
            env_num (int): the index of the single environment

        Returns:
            infos (dict): the (updated) infos of the vectorized environment

        """
        for k in info.keys():
            if k not in infos:
                info_array, array_mask = self._init_info_arrays(type(info[k]))
            else:
                info_array, array_mask = infos[k], infos[f"_{k}"]

            info_array[env_num], array_mask[env_num] = info[k], True
            infos[k], infos[f"_{k}"] = info_array, array_mask
        return infos

    def _init_info_arrays(
        self, dtype: type
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
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
            array = np.zeros(self.num_envs, dtype=object)
            array[:] = None
        array_mask = np.zeros(self.num_envs, dtype=np.bool_)
        return array, array_mask

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
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"


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
        """Initialises the wrapper with :attr:`env` to wrap and an observation and action space."""
        super().__init__()

        assert isinstance(env, VectorEnv)
        self.env = env

        self._action_space: spaces.Space[ActType] | None = None
        self._observation_space: spaces.Space[ObsType] | None = None
        self._metadata: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VectorWrapperObsType, dict[str, Any]]:
        """Calls :meth:`reset` on the :attr:`env` allowing for modification of observations, info or options."""
        return self.env.reset(seed=seed, options=options)

    def step(
        self, actions: VectorWrapperActType
    ) -> tuple[
        VectorWrapperObsType,
        VectorWrapperArrayType,
        VectorWrapperArrayType,
        VectorWrapperArrayType,
        dict[str, Any],
    ]:
        """Calls :meth:`step` on the :attr:`env` allowing for modification of actions, observations, etc."""
        return self.env.step(actions)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.render()

    def close(self, **kwargs):
        """Calls :meth:`close` on the :attr:`env`."""
        return self.env.close()

    def __getattr__(self, name: str) -> Any:
        """If the wrapper does not contain the attribute, we try to access the :attr:`env` with the attribute."""
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    @property
    def spec(self) -> EnvSpec | None:
        """Returns the :attr:`Env` :attr:`spec` attribute."""
        return self.env.spec

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def action_space(
        self,
    ) -> spaces.Space[VectorActType] | spaces.Space[VectorWrapperActType]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space[VectorWrapperActType]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> spaces.Space[VectorObsType] | spaces.Space[VectorWrapperObsType]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space[VectorWrapperActType]):
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
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the :attr:`Env` :attr:`np_random` attribute."""
        return self.env.np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self.env.np_random = value

    @property
    def _np_random(self):
        raise AttributeError(
            "Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`."
        )

    @property
    def unwrapped(self) -> VectorEnv[VectorObsType, VectorActType, VectorArrayType]:
        """Returns the base :class:`VectorEnv`."""
        return self.env.unwrapped

    def __repr__(self):
        """Returns a representation of the vector wrapper."""
        return f"<{self.__class__.__name__}, {self.env}>"

    def __del__(self):
        """Calls :meth:`__del__` on the :attr:`env`."""
        self.env.__del__()


class VectorObservationWrapper(
    VectorWrapper[VectorWrapperObsType, VectorActType, VectorArrayType]
):
    """Wraps the vectorized environment to allow a modular transformation of the observation. Equivalent to :class:`gym.ObservationWrapper` for vectorized environments."""

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VectorWrapperObsType, dict[str, Any]]:
        """Calls :meth:`reset` using the :attr:`env` and modifies the returned observation using :meth:`observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, actions: VectorActType
    ) -> tuple[
        VectorWrapperObsType,
        VectorArrayType,
        VectorArrayType,
        VectorArrayType,
        dict[str, Any],
    ]:
        """Calls :meth:`step` using the :attr:`env` and modifies the returned observation using :meth:`observation`."""
        obs, rewards, terminations, truncations, info = self.env.step(actions)
        return (
            self.observation(obs),
            rewards,
            terminations,
            truncations,
            info,
        )

    def observation(self, observation: VectorObsType) -> VectorWrapperObsType:
        """Defines the observation transformation.

        Args:
            observation (VectorObsType): the observation from the environment

        Returns:
            observation (VectorWrapperObsType): the transformed observation
        """
        raise NotImplementedError


class VectorActionWrapper(
    VectorWrapper[VectorObsType, VectorWrapperActType, VectorArrayType]
):
    """Wraps the vectorized environment to allow a modular transformation of the actions. Equivalent of :class:`~gym.ActionWrapper` for vectorized environments."""

    def step(self, actions: VectorWrapperActType):
        """Calls :meth:`step` using the :attr:`env` with a modified action from :meth:`actions`."""
        return self.env.step(self.action(actions))

    def actions(self, actions: VectorWrapperActType) -> VectorActType:
        """Transform the actions before sending them to the environment.

        Args:
            actions (ActType): the actions to transform

        Returns:
            ActType: the transformed actions
        """
        raise NotImplementedError


class VectorRewardWrapper(VectorWrapper[VectorObsType, VectorActType, VectorArrayType]):
    """Wraps the vectorized environment to allow a modular transformation of the reward. Equivalent of :class:`~gym.RewardWrapper` for vectorized environments."""

    def step(
        self, actions: VectorActType
    ) -> tuple[
        VectorObsType,
        VectorArrayType,
        VectorArrayType,
        VectorArrayType,
        dict[str, Any],
    ]:
        """Calls :meth:`step` on the :attr:`env` modifying the reward using :meth:`reward`."""
        observation, reward, termination, truncation, info = self.env.step(actions)
        return observation, self.reward(reward), termination, truncation, info

    def reward(self, reward: VectorArrayType) -> VectorArrayType:
        """Transform the reward before returning it.

        Args:
            reward (array): the reward to transform

        Returns:
            array: the transformed reward
        """
        raise NotImplementedError
