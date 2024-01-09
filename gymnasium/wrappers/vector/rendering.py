"""A collections of rendering-based wrappers.

* ``HumanRendering`` - Provides human rendering of vector environments with ``"rgb_array"``
"""
from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType


__all__ = [
    "HumanRendering",
]

from gymnasium.wrappers.base_.rendering import HumanRenderingBase


class HumanRendering(
    gym.vector.VectorWrapper,
    HumanRenderingBase[
        gym.vector.VectorWrapper,
        gym.vector.VectorEnv[ObsType, ActType, gym.vector.vector_env.ArrayType],
    ],
    gym.utils.RecordConstructorArgs,
):
    # noinspection PyShadowingNames
    """Allows human like rendering for vector environments that support "rgb_array" rendering.

    This wrapper is particularly useful when you have implemented an environment that can produce
    RGB images but haven't implemented any code to render the images to the screen.
    If you want to use this wrapper with your environments, remember to specify ``"render_fps"``
    in the metadata of your environment.

    The ``render_mode`` of the wrapped environment must be either ``'rgb_array'`` or ``'rgb_array_list'``.

    Example:
        >>> from gymnasium import make_vec
        >>> from gymnasium.wrappers.vector.rendering import HumanRendering
        >>> env = make_vec("CartPole-v1", num_envs=3, vector_kwargs=dict(render_mode="rgb_array"))
        >>> wrapped = HumanRendering(env)  # Will warn that ChartPole-v1 natively supports 'human' rendering
        >>> obs, _ = wrapped.reset()     # This will start rendering to the screen
        >>> wrapped.render_mode, env.render_mode, env.unwrapped.render_mode  # extra check
        ('human', 'rgb_array', 'rgb_array')

        Warning: If the base environment uses ``render_mode="rgb_array_list"``, its (i.e. the *base environment's*)
         render method will always return an empty list:

    Change logs:
     * v1.0.0 - Initially added
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv[ObsType, ActType, gym.vector.vector_env.ArrayType],
    ):
        """Initialize a :class:`HumanRendering` instance.

        Note: RenderState can determine if only one process should be rendered or multiple

        Args:
            env: The environment that is being wrapped
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.vector.VectorWrapper.__init__(self, env)  # Creates the 'unwrapped' property
        HumanRenderingBase.__init__(self, env, num_envs=self.num_envs)

    @property
    def render_mode(self):
        """Returns 'human' when env has a compatible mode."""
        return self._get_render_mode(self.env)

    @render_mode.setter
    def render_mode(self, mode: str):
        """Forwards render mode other than 'human' to env."""
        self._set_render_mode(self.env, mode)

    @property
    def autoreset_envs(self):
        """Make wrapped :attr:`autoreset_envs` readable."""
        return self.unwrapped.autoreset_envs

    @autoreset_envs.setter
    def autoreset_envs(self, value) -> None:
        """Make wrapped :attr:`autoreset_envs` writeable."""
        self.unwrapped.autoreset_envs = value

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        """Perform a step in the base environment and render a frame to the screen."""
        result = super().step(action)
        self._render_frame()
        return result

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the base environment and render a frame to the screen."""
        result = super().reset(seed=seed, options=options)
        self._check_config(self.env)
        self._render_frame()
        return result

    def render(self) -> None:
        """This method doesn't do much, actual rendering is performed in :meth:`step` and :meth:`reset`."""
        return None

    def close(self):
        """Close the rendering window."""
        self._close()
        super().close()
