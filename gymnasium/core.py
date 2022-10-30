"""Core API for Environment, Wrapper, ActionWrapper, RewardWrapper and ObservationWrapper."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class Env(Generic[ObsType, ActType]):
    r"""The main Gymnasium class for implementing Reinforcement Learning Agents environments.

    The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the :meth:`step` and :meth:`reset` functions.
    An environment can be partially or fully observed by single agents. For multi-agent environments, see PettingZoo.

    The main API methods that users of this class need to know are:

    - :meth:`step` - Updates an environment with actions returning the next agent observation, the reward for taking that actions,
      if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
    - :meth:`reset` - Resets the environment to an initial state, required before calling step.
      Returns the first agent observation for an episode and information, i.e. metrics, debug info.
    - :meth:`render` - Renders the environments to help visualise what the agent see, examples modes are "human", "rgb_array", "ansi" for text.
    - :meth:`close` - Closes the environment, important when external software is used, i.e. pygame for rendering, databases

    Environments have additional attributes for users to understand the implementation

    - :attr:`action_space` - The Space object corresponding to valid actions, all valid actions should be contained within the space.
    - :attr:`observation_space` - The Space object corresponding to valid observations, all valid observations should be contained within the space.
    - :attr:`reward_range` - A tuple corresponding to the minimum and maximum possible rewards for an agent over an episode.
      The default reward range is set to :math:`(-\infty,+\infty)`.
    - :attr:`spec` - An environment spec that contains the information used to initialize the environment from :meth:`gymnasium.make`
    - :attr:`metadata` - The metadata of the environment, i.e. render modes, render fps
    - :attr:`np_random` - The random number generator for the environment. This is automatically assigned during
      ``super().reset(seed=seed)`` and when assessing ``self.np_random``.

    .. seealso:: For modifying or extending environments use the :py:class:`gymnasium.Wrapper` class
    """

    # Set this in SOME subclasses
    metadata: dict[str, Any] = {"render_modes": []}
    # define render_mode if your environment supports rendering
    render_mode: str | None = None
    reward_range = (-float("inf"), float("inf"))
    spec: EnvSpec | None = None

    # Set these in ALL subclasses
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    # Created
    _np_random: np.random.Generator | None = None

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        .. versionchanged:: 0.26

            The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
            to users when the environment had terminated or truncated which is critical for reinforcement learning
            bootstrapping algorithms.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged:: v0.25

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        The environment's :attr:`metadata` render modes (`env.metadata["render_modes"]`) should contain the possible
        ways to implement the render modes. In addition, list versions for most render modes is achieved through
        `gymnasium.make` which automatically applies a wrapper to collect rendered frames.

        Note:
            As the :attr:`render_mode` is known during ``__init__``, the objects used to render the environment state
            should be initialised in ``__init__``.

        By convention, if the :attr:`render_mode` is:

        - None (default): no render is computed.
        - "human": The environment is continuously rendered in the current display or terminal, usually for human consumption.
          This rendering should occur during :meth:`step` and :meth:`render` doesn't need to be called. Returns ``None``.
        - "rgb_array": Return a single frame representing the current state of the environment.
          A frame is a ``np.ndarray`` with shape ``(x, y, 3)`` representing RGB values for an x-by-y pixel image.
        - "ansi": Return a strings (``str``) or ``StringIO.StringIO`` containing a terminal-style text representation
          for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
        - "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human) through the
          wrapper, :py:class:`gymnasium.wrappers.RenderCollection` that is automatically applied during ``gymnasium.make(..., render_mode="rgb_array_list")``.
          The frames collected are popped after :meth:`render` is called or :meth:`reset`.

        Note:
            Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes the list of supported modes.

        .. versionchanged:: 0.25.0

            The render function was changed to no longer accept parameters, rather these parameters should be specified
            in the environment initialised, i.e., ``gymnasium.make("CartPole-v1", render_mode="human")``
        """
        raise NotImplementedError

    def close(self):
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        """
        pass

    @property
    def unwrapped(self) -> Env[ObsType, ActType]:
        """Returns the base non-wrapped environment.

        Returns:
            Env: The base non-wrapped :class:`gymnasium.Env` instance
        """
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

    def __str__(self):
        """Returns a string of the environment with :attr:`spec` id's if :attr:`spec.

        Returns:
            A string identifying the environment
        """
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any):
        """Support with-statement for the environment and closes the environment."""
        self.close()
        # propagate exception
        return False


WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class Wrapper(Env[WrapperObsType, WrapperActType]):
    """Wraps a :class:`gymnasium.Env` to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

    This class is the base class of all wrappers to change the behavior of the underlying environment allowing
    modification to the :attr:`action_space`, :attr:`observation_space`, :attr:`reward_range` and :attr:`metadata`
    that doesn't change the underlying environment attributes.

    In addition, for several attributes (:attr:`spec`, :attr:`render_mode`, :attr:`np_random`) will point back to the
    wrapper's environment.

    Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.
    Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular. Wrappers can
    also be chained to combine their effects. Most environments that are generated via `gymnasium.make` will already be wrapped by default.

    In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along
    with (possibly optional) parameters to the wrapper's constructor.

        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import RescaleAction
        >>> base_env = gym.make("BipedalWalker-v3")
        >>> base_env.action_space
        Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)
        >>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
        >>> wrapped_env.action_space
        Box([0. 0. 0. 0.], [1. 1. 1. 1.], (4,), float32)

    You can access the environment underneath the **first** wrapper by using the :attr:`env` attribute.
    As the :class:`Wrapper` class inherits from :class:`Env` then :attr:`env` can be another wrapper.

        >>> wrapped_env
        <RescaleAction<TimeLimit<OrderEnforcing<BipedalWalker<BipedalWalker-v3>>>>>
        >>> wrapped_env.env
        <TimeLimit<OrderEnforcing<BipedalWalker<BipedalWalker-v3>>>>

    If you want to get to the environment underneath **all** of the layers of wrappers, you can use the `.unwrapped` attribute.
    If the environment is already a bare environment, the `.unwrapped` attribute will just return itself.

        >>> wrapped_env
        <RescaleAction<TimeLimit<OrderEnforcing<BipedalWalker<BipedalWalker-v3>>>>>
        >>> wrapped_env.unwrapped
        <gymnasium.envs.box2d.bipedal_walker.BipedalWalker object at 0x7f87d70712d0>

    There are three common things you might want a wrapper to do:

    - Transform actions before applying them to the base environment
    - Transform observations that are returned by the base environment
    - Transform rewards that are returned by the base environment

    Such wrappers can be easily implemented by inheriting from `ActionWrapper`, `ObservationWrapper`, or `RewardWrapper` and implementing the
    respective transformation. If you need a wrapper to do more complicated tasks, you can inherit from the `Wrapper` class directly.
    The code that is presented in the following sections can also be found in
    the [gym-examples](https://github.com/Farama-Foundation/gym-examples) repository

    Note:
        Don't forget to call ``super().__init__(env)``
    """

    def __init__(self, env: Env[ObsType, ActType]):
        """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

        Args:
            env: The environment to wrap
        """
        self.env = env

        self._action_space: spaces.Space[WrapperActType] | None = None
        self._observation_space: spaces.Space[WrapperObsType] | None = None
        self._reward_range: tuple[SupportsFloat, SupportsFloat] | None = None
        self._metadata: dict[str, Any] | None = None

    def __getattr__(self, name: str):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
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
    ) -> spaces.Space[ActType] | spaces.Space[WrapperActType]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space[WrapperActType]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> spaces.Space[ObsType] | spaces.Space[WrapperObsType]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space[WrapperObsType]):
        self._observation_space = space

    @property
    def reward_range(self) -> tuple[SupportsFloat, SupportsFloat]:
        """Return the :attr:`Env` :attr:`reward_range` unless overwritten then the wrapper :attr:`reward_range` is used."""
        if self._reward_range is None:
            return self.env.reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value: tuple[SupportsFloat, SupportsFloat]):
        self._reward_range = value

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
        """This code will never be run due to __getattr__ being called prior this.

        It seems that @property overwrites the variable (`_np_random`) meaning that __getattr__ gets called with the missing variable.
        """
        raise AttributeError(
            "Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`."
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.render()

    def close(self):
        """Closes the wrapper and :attr:`env`."""
        return self.env.close()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    @property
    def unwrapped(self) -> Env[ObsType, ActType]:
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped


class ObservationWrapper(Wrapper[WrapperObsType, ActType]):
    """Superclass of wrappers that can modify observations using :meth:`observation` for :meth:`reset` and :meth:`step`.

    If you would like to apply a function to only the observation before
    passing it to the learning code, you can simply inherit from :class:`ObservationWrapper` and overwrite the method
    :meth:`observation` to implement that transformation. The transformation defined in that method must be
    reflected by the :attr:`env` observation space. Otherwise, you need to specify the new observation space of the
    wrapper by setting :attr:`self.observation_space` in the :meth:`__init__` method of your wrapper.

    For example, you might have a 2D navigation task where the environment returns dictionaries as observations with
    keys ``"agent_position"`` and ``"target_position"``. A common thing to do might be to throw away some degrees of
    freedom and only consider the position of the target relative to the agent, i.e.
    ``observation["target_position"] - observation["agent_position"]``. For this, you could implement an
    observation wrapper like this::

        class RelativePosition(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

            def observation(self, obs):
                return obs["target"] - obs["agent"]

    Among others, Gymnasium provides the observation wrapper :class:`TimeAwareObservation`, which adds information about the
    index of the timestep to the observation.
    """

    def __init__(self, env: Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        raise NotImplementedError


class RewardWrapper(Wrapper[ObsType, ActType]):
    """Superclass of wrappers that can modify the returning reward from a step.

    If you would like to apply a function to the reward that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`RewardWrapper` and overwrite the method
    :meth:`reward` to implement that transformation.
    This transformation might change the :attr:`reward_range`; to specify the :attr:`reward_range` of your wrapper,
    you can simply define :attr:`self.reward_range` in :meth:`__init__`.

    Let us look at an example: Sometimes (especially when we do not have control over the reward
    because it is intrinsic), we want to clip the reward to a range to gain some numerical stability.
    To do that, we could, for instance, implement the following wrapper::

        class ClipReward(gym.RewardWrapper):
            def __init__(self, env, min_reward, max_reward):
                super().__init__(env)
                self.min_reward = min_reward
                self.max_reward = max_reward
                self.reward_range = (min_reward, max_reward)

            def reward(self, r: SupportsFloat) -> SupportsFloat:
                return np.clip(r, self.min_reward, self.max_reward)
    """

    def __init__(self, env: Env[ObsType, ActType]):
        """Constructor for the Reward wrapper."""
        super().__init__(env)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward), terminated, truncated, info

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        raise NotImplementedError


class ActionWrapper(Wrapper[ObsType, WrapperActType]):
    """Superclass of wrappers that can modify the action before :meth:`env.step`.

    If you would like to apply a function to the action before passing it to the base environment,
    you can simply inherit from :class:`ActionWrapper` and overwrite the method :meth:`action` to implement
    that transformation. The transformation defined in that method must take values in the base environment’s
    action space. However, its domain might differ from the original action space.
    In that case, you need to specify the new action space of the wrapper by setting :attr:`self.action_space` in
    the :meth:`__init__` method of your wrapper.

    Let’s say you have an environment with action space of type :class:`gymnasium.spaces.Box`, but you would only like
    to use a finite subset of actions. Then, you might want to implement the following wrapper::

        class DiscreteActions(gym.ActionWrapper):
            def __init__(self, env, disc_to_cont):
                super().__init__(env)
                self.disc_to_cont = disc_to_cont
                self.action_space = Discrete(len(disc_to_cont))

            def action(self, act):
                return self.disc_to_cont[act]

        if __name__ == "__main__":
            env = gym.make("LunarLanderContinuous-v2")
            wrapped_env = DiscreteActions(env, [np.array([1,0]), np.array([-1,0]),
                                                np.array([0,1]), np.array([0,-1])])
            print(wrapped_env.action_space)         #Discrete(4)

    Among others, Gymnasium provides the action wrappers :class:`ClipAction` and :class:`RescaleAction` for clipping and rescaling actions.
    """

    def __init__(self, env: Env[ObsType, ActType]):
        """Constructor for the action wrapper."""
        super().__init__(env)

    def step(
        self, action: WrapperActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        return self.env.step(self.action(action))

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        raise NotImplementedError
