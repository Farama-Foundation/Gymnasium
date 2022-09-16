"""Core API for Environment, Wrapper, ActionWrapper, RewardWrapper and ObservationWrapper."""
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from gymnasium import spaces
from gymnasium.logger import warn
from gymnasium.utils import seeding

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec

if sys.version_info[0:2] == (3, 6):
    warn(
        "Gymnasium minimally supports python 3.6 as the python foundation not longer supports the version, please update your version to 3.7+"
    )

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class Env(Generic[ObsType, ActType]):
    r"""The main Gymnasium class.

    It encapsulates an environment with arbitrary behind-the-scenes dynamics.
    An environment can be partially or fully observed.

    The main API methods that users of this class need to know are:

    - :meth:`step` - Takes a step in the environment using an action returning the next observation, reward,
      if the environment terminated and observation information.
    - :meth:`reset` - Resets the environment to an initial state, returning the initial observation and observation information.
    - :meth:`render` - Renders the environment observation with modes depending on the output
    - :meth:`close` - Closes the environment, important for rendering where pygame is imported

    And set the following attributes:

    - :attr:`action_space` - The Space object corresponding to valid actions
    - :attr:`observation_space` - The Space object corresponding to valid observations
    - :attr:`reward_range` - A tuple corresponding to the minimum and maximum possible rewards
    - :attr:`spec` - An environment spec that contains the information used to initialise the environment from `gym.make`
    - :attr:`metadata` - The metadata of the environment, i.e. render modes
    - :attr:`np_random` - The random number generator for the environment

    Note: a default reward range set to :math:`(-\infty,+\infty)` already exists. Set it if you want a narrower range.
    """

    # Set this in SOME subclasses
    metadata: Dict[str, Any] = {"render_modes": []}
    # define render_mode if your environment supports rendering
    render_mode: Optional[str] = None
    reward_range = (-float("inf"), float("inf"))
    spec: "EnvSpec" = None

    # Set these in ALL subclasses
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    # Created
    _np_random: Optional[np.random.Generator] = None

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed."""
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(observation, reward, terminated, truncated, info)`.

        Args:
            action (ActType): an action provided by the agent

        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """Resets the environment to an initial state and returns the initial observation.

        This method can reset the environment's random number generator(s) if ``seed`` is an integer or
        if the environment has not yet initialized a random number generator.
        If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
        the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG.
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)


        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Compute the render frames as specified by render_mode attribute during initialization of the environment.

        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if render_mode is:

        - None (default): no render is computed.
        - human: render return None.
          The environment is continuously rendered in the current display or terminal. Usually for human consumption.
        - rgb_array: return a single frame representing the current state of the environment.
          A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
        - rgb_array_list: return a list of frames representing the states of the environment since the last reset.
          Each frame is a numpy.ndarray with shape (x, y, 3), as with `rgb_array`.
        - ansi: Return a strings (str) or StringIO.StringIO containing a
          terminal-style text representation for each time step.
          The text can include newlines and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render_modes' key includes
            the list of supported modes. It's recommended to call super()
            in implementations to use the functionality of this method.
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        pass

    @property
    def unwrapped(self) -> "Env":
        """Returns the base non-wrapped environment.

        Returns:
            Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        """Returns a string of the environment with the spec id if specified."""
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False


class Wrapper(Env[ObsType, ActType]):
    """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    Note:
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """

    def __init__(self, env: Env):
        """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

        Args:
            env: The environment to wrap
        """
        self.env = env

        self._action_space: Optional[spaces.Space] = None
        self._observation_space: Optional[spaces.Space] = None
        self._reward_range: Optional[Tuple[SupportsFloat, SupportsFloat]] = None
        self._metadata: Optional[dict] = None

    def __getattr__(self, name):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    @property
    def spec(self):
        """Returns the environment specification."""
        return self.env.spec

    @classmethod
    def class_name(cls):
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def action_space(self) -> spaces.Space[ActType]:
        """Returns the action space of the environment."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space):
        self._action_space = space

    @property
    def observation_space(self) -> spaces.Space:
        """Returns the observation space of the environment."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space):
        self._observation_space = space

    @property
    def reward_range(self) -> Tuple[SupportsFloat, SupportsFloat]:
        """Return the reward range of the environment."""
        if self._reward_range is None:
            return self.env.reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value: Tuple[SupportsFloat, SupportsFloat]):
        self._reward_range = value

    @property
    def metadata(self) -> dict:
        """Returns the environment metadata."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def render_mode(self) -> Optional[str]:
        """Returns the environment render_mode."""
        return self.env.render_mode

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment np_random."""
        return self.env.np_random

    @np_random.setter
    def np_random(self, value):
        self.env.np_random = value

    @property
    def _np_random(self):
        raise AttributeError(
            "Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`."
        )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps through the environment with action."""
        return self.env.step(action)

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        """Resets the environment with kwargs."""
        return self.env.reset(**kwargs)

    def render(
        self, *args, **kwargs
    ) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Renders the environment."""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Closes the environment."""
        return self.env.close()

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    @property
    def unwrapped(self) -> Env:
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped


class ObservationWrapper(Wrapper):
    """Superclass of wrappers that can modify observations using :meth:`observation` for :meth:`reset` and :meth:`step`.

    If you would like to apply a function to the observation that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`ObservationWrapper` and overwrite the method
    :meth:`observation` to implement that transformation. The transformation defined in that method must be
    defined on the base environment’s observation space. However, it may take values in a different space.
    In that case, you need to specify the new observation space of the wrapper by setting :attr:`self.observation_space`
    in the :meth:`__init__` method of your wrapper.

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

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError


class RewardWrapper(Wrapper):
    """Superclass of wrappers that can modify the returning reward from a step.

    If you would like to apply a function to the reward that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`RewardWrapper` and overwrite the method
    :meth:`reward` to implement that transformation.
    This transformation might change the reward range; to specify the reward range of your wrapper,
    you can simply define :attr:`self.reward_range` in :meth:`__init__`.

    Let us look at an example: Sometimes (especially when we do not have control over the reward
    because it is intrinsic), we want to clip the reward to a range to gain some numerical stability.
    To do that, we could, for instance, implement the following wrapper::

        class ClipReward(gymnasium.RewardWrapper):
            def __init__(self, env, min_reward, max_reward):
                super().__init__(env)
                self.min_reward = min_reward
                self.max_reward = max_reward
                self.reward_range = (min_reward, max_reward)

            def reward(self, reward):
                return np.clip(reward, self.min_reward, self.max_reward)
    """

    def step(self, action):
        """Modifies the reward using :meth:`self.reward` after the environment :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        """Returns a modified ``reward``."""
        raise NotImplementedError


class ActionWrapper(Wrapper):
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


    Among others, Gymnasium provides the action wrappers :class:`ClipAction` and :class:`RescaleAction`.
    """

    def step(self, action):
        """Runs the environment :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        return self.env.step(self.action(action))

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""
        raise NotImplementedError

    def reverse_action(self, action):
        """Returns a reversed ``action``."""
        raise NotImplementedError
