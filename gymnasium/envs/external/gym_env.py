from functools import singledispatch
from typing import Optional, Tuple

import gymnasium
from gymnasium.core import ActType, ObsType

try:
    import gym
except ImportError as e:
    # gym = None
    GYM_IMPORT_ERROR = e
else:
    GYM_IMPORT_ERROR = None


class GymEnvironment(gymnasium.Env):
    """ """

    def __init__(self, env_id: str, make_kwargs: Optional[dict] = None):
        if GYM_IMPORT_ERROR is not None:
            raise gymnasium.error.DependencyNotInstalled(
                f"{GYM_IMPORT_ERROR} (Hint: You need to install gym with `pip install gym` to use gym environments"
            )

        if make_kwargs is None:
            make_kwargs = {}

        self.gym_env = _strip_default_wrappers(gym.make(env_id, **make_kwargs))

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
        super().reset(seed=None)  # We don't need the seed inside gymnasium
        # Options are ignored
        return self.gym_env.reset()

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
    import gym.wrappers.compatibility  # Cheat because gym doesn't expose it in __init__
    import gym.wrappers.env_checker  # Cheat because gym doesn't expose it in __init__

    default_wrappers = (
        gym.wrappers.RenderCollection,
        gym.wrappers.HumanRendering,
        gym.wrappers.AutoResetWrapper,
        gym.wrappers.TimeLimit,
        gym.wrappers.OrderEnforcing,
        gym.wrappers.env_checker.PassiveEnvChecker,
        gym.wrappers.compatibility.EnvCompatibility,
    )
    while isinstance(env, default_wrappers):
        env = env.env
    return env


@singledispatch
def _convert_space(space: "gym.Space") -> gymnasium.Space:
    """Blah"""
    raise NotImplementedError(
        f"Cannot convert space of type {type(space)}. Please upgrade your code to gymnasium."
    )


@_convert_space.register
def _(space: "gym.spaces.Discrete") -> gymnasium.spaces.Discrete:
    return gymnasium.spaces.Discrete(space.n)


@_convert_space.register
def _(space: "gym.spaces.Box") -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(space.low, space.high, space.shape)


@_convert_space.register
def _(space: "gym.spaces.Tuple") -> gymnasium.spaces.Tuple:
    return gymnasium.spaces.Tuple(_convert_space(s) for s in space.spaces)


@_convert_space.register
def _(space: "gym.spaces.Dict") -> gymnasium.spaces.Dict:
    return gymnasium.spaces.Dict(
        {k: _convert_space(s) for k, s in space.spaces.items()}
    )


@_convert_space.register
def _(space: "gym.spaces.MultiDiscrete") -> gymnasium.spaces.MultiDiscrete:
    return gymnasium.spaces.MultiDiscrete(space.nvec)


@_convert_space.register
def _(space: "gym.spaces.MultiBinary") -> gymnasium.spaces.MultiBinary:
    return gymnasium.spaces.MultiBinary(space.n)


@_convert_space.register
def _(space: "gym.spaces.Sequence") -> gymnasium.spaces.Sequence:
    return gymnasium.spaces.Sequence(_convert_space(space.feature_space))


@_convert_space.register
def _(space: "gym.spaces.Graph") -> gymnasium.spaces.Graph:
    # Pycharm is throwing up a type warning, but as long as the base space is correct, this is valid
    return gymnasium.spaces.Graph(_convert_space(space.node_space), _convert_space(space.edge_space))  # type: ignore
