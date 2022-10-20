from typing import Optional, Tuple

import gymnasium
from gymnasium import error
from gymnasium.core import ActType, ObsType

try:
    import gym
    import gym.wrappers
except ImportError as e:
    GYM_IMPORT_ERROR = e
else:
    GYM_IMPORT_ERROR = None


class GymEnvironment(gymnasium.Env):
    """
    Converts a gym environment to a gymnasium environment.
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        make_kwargs: Optional[dict] = None,
        env: Optional["gym.Env"] = None,
    ):
        if GYM_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(
                f"{GYM_IMPORT_ERROR} (Hint: You need to install gym with `pip install gym` to use gym environments"
            )

        if make_kwargs is None:
            make_kwargs = {}

        if env is not None:
            self.gym_env = env
        elif env_id is not None:
            self.gym_env = gym.make(env_id, **make_kwargs)
        else:
            raise gymnasium.error.MissingArgument(
                "Either env_id or env must be provided to create a legacy gym environment."
            )
        self.gym_env = _strip_default_wrappers(self.gym_env)

        self.observation_space = _convert_space(self.gym_env.observation_space)
        self.action_space = _convert_space(self.gym_env.action_space)

        self.metadata = getattr(self.gym_env, "metadata", {"render_modes": []})
        self.render_mode = self.gym_env.render_mode
        self.reward_range = getattr(self.gym_env, "reward_range", None)
        self.spec = getattr(self.gym_env, "spec", None)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        # Options are ignored
        return self.gym_env.reset(seed=seed, options=options)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        return self.gym_env.step(action)

    def render(self):
        """Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        return self.gym_env.render()

    def close(self):
        """Closes the environment."""
        self.gym_env.close()

    def __str__(self):
        return f"GymEnvironment({self.gym_env})"

    def __repr__(self):
        return f"GymEnvironment({self.gym_env})"


def _strip_default_wrappers(env: "gym.Env") -> "gym.Env":
    """Strips builtin wrappers from the environment.

    Args:
        env: the environment to strip builtin wrappers from

    Returns:
        The environment without builtin wrappers
    """

    default_wrappers = (
        gym.wrappers.render_collection.RenderCollection,
        gym.wrappers.human_rendering.HumanRendering,
    )
    while isinstance(env, default_wrappers):
        env = env.env
    return env


def _convert_space(space: "gym.Space") -> gymnasium.Space:
    """Converts a gym space to a gymnasium space.

    Args:
        space: the space to convert

    Returns:
        The converted space
    """
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low, high=space.high, shape=space.shape, dtype=space.dtype
        )
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return gymnasium.spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return gymnasium.spaces.MultiBinary(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return gymnasium.spaces.Tuple(spaces=tuple(map(_convert_space, space.spaces)))
    elif isinstance(space, gym.spaces.Dict):
        return gymnasium.spaces.Dict(
            spaces={k: _convert_space(v) for k, v in space.spaces.items()}
        )
    elif isinstance(space, gym.spaces.Sequence):
        return gymnasium.spaces.Sequence(space=_convert_space(space.feature_space))
    elif isinstance(space, gym.spaces.Graph):
        return gymnasium.spaces.Graph(
            node_space=_convert_space(space.node_space),  # type: ignore
            edge_space=_convert_space(space.edge_space),  # type: ignore
        )
    elif isinstance(space, gym.spaces.Text):
        return gymnasium.spaces.Text(
            max_length=space.max_length,
            min_length=space.min_length,
            charset=space._char_str,
        )
    else:
        raise NotImplementedError(
            f"Cannot convert space of type {space}. Please upgrade your code to gymnasium."
        )
