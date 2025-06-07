"""Implementation of a Jax-accelerated pendulum environment."""

from __future__ import annotations

from os import path
from typing import Any, Optional, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

import gymnasium as gym
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv, FunctionalJaxVectorEnv
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import ActType, FuncEnv
from gymnasium.utils import EzPickle
from gymnasium.vector import AutoresetMode


PRNGKeyType: TypeAlias = jax.Array
StateType: TypeAlias = jax.Array
RenderStateType = tuple["pygame.Surface", "pygame.time.Clock", Optional[float]]  # type: ignore  # noqa: F821


@struct.dataclass
class PendulumParams:
    """Parameters for the jax Pendulum environment."""

    max_speed: float = 8.0
    dt: float = 0.05
    g: float = 10.0
    m: float = 1.0
    l: float = 1.0
    high_x: float = jnp.pi
    high_y: float = 1.0
    screen_dim: int = 500


class PendulumFunctional(
    FuncEnv[StateType, jax.Array, int, float, bool, RenderStateType, PendulumParams]
):
    """Pendulum but in jax and functional structure."""

    max_torque: float = 2.0

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-max_torque, max_torque, shape=(1,), dtype=np.float32)

    def initial(
        self, rng: PRNGKeyType, params: PendulumParams = PendulumParams
    ) -> StateType:
        """Initial state generation."""
        high = jnp.array([params.high_x, params.high_y])
        return jax.random.uniform(key=rng, minval=-high, maxval=high, shape=high.shape)

    def transition(
        self,
        state: StateType,
        action: int | jax.Array,
        rng: None = None,
        params: PendulumParams = PendulumParams,
    ) -> StateType:
        """Pendulum transition."""
        th, thdot = state  # th := theta
        u = action

        g = params.g
        m = params.m
        l = params.l
        dt = params.dt

        u = jnp.clip(u, -self.max_torque, self.max_torque)[0]

        newthdot = thdot + (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)
        newth = th + newthdot * dt

        new_state = jnp.array([newth, newthdot])
        return new_state

    def observation(
        self, state: StateType, rng: Any, params: PendulumParams = PendulumParams
    ) -> jax.Array:
        """Generates an observation based on the state."""
        theta, thetadot = state
        return jnp.array([jnp.cos(theta), jnp.sin(theta), thetadot])

    def reward(
        self,
        state: StateType,
        action: ActType,
        next_state: StateType,
        rng: Any,
        params: PendulumParams = PendulumParams,
    ) -> float:
        """Generates the reward based on the state, action and next state."""
        th, thdot = state  # th := theta
        u = action

        u = jnp.clip(u, -self.max_torque, self.max_torque)[0]

        th_normalized = ((th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        costs = th_normalized**2 + 0.1 * thdot**2 + 0.001 * (u**2)

        return -costs

    def terminal(
        self, state: StateType, rng: Any, params: PendulumParams = PendulumParams
    ) -> bool:
        """Determines if the state is a terminal state."""
        return False

    def render_image(
        self,
        state: StateType,
        render_state: RenderStateType,
        params: PendulumParams = PendulumParams,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Renders an RGB image."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        screen, clock, last_u = render_state

        surf = pygame.Surface((params.screen_dim, params.screen_dim))
        surf.fill((255, 255, 255))

        bound = 2.2
        scale = params.screen_dim / (bound * 2)
        offset = params.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(last_u) / 2, scale * np.abs(last_u) / 2),
            )
            is_flip = bool(last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        gfxdraw.aacircle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        return (screen, clock, last_u), np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

    def render_init(
        self,
        screen_width: int = 600,
        screen_height: int = 400,
        params: PendulumParams = PendulumParams,
    ) -> RenderStateType:
        """Initialises the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        pygame.init()
        screen = pygame.Surface((screen_width, screen_height))
        clock = pygame.time.Clock()

        return screen, clock, None

    def render_close(
        self,
        render_state: RenderStateType,
        params: PendulumParams = PendulumParams,
    ):
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        pygame.display.quit()
        pygame.quit()

    def get_default_params(self, **kwargs) -> PendulumParams:
        """Returns the default parameters for the environment."""
        return PendulumParams(**kwargs)


class PendulumJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based pendulum environment using the functional version as base."""

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
        "jax": True,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(self, render_mode: str | None = None, **kwargs: Any):
        """Constructor where the kwargs are passed to the base environment to modify the parameters."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        env = PendulumFunctional(**kwargs)
        env.transform(jax.jit)

        super().__init__(
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class PendulumJaxVectorEnv(FunctionalJaxVectorEnv, EzPickle):
    """Jax-based implementation of the vectorized CartPole environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(
        self,
        num_envs: int,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        **kwargs: Any,
    ):
        """Constructor for the vectorized CartPole where the kwargs are applied to the functional environment."""
        EzPickle.__init__(
            self,
            num_envs=num_envs,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        env = PendulumFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxVectorEnv.__init__(
            self,
            func_env=env,
            num_envs=num_envs,
            metadata=self.metadata,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
