"""Implementation of a Jax-accelerated cartpole environment."""

from __future__ import annotations

from typing import Any, TypeAlias

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
RenderStateType = tuple["pygame.Surface", "pygame.time.Clock"]  # type: ignore  # noqa: F821


@struct.dataclass
class CartPoleParams:
    """Parameters for the jax CartPole environment."""

    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = masspole + masscart
    length: float = 0.5
    polemass_length: float = masspole + length
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12 * 2 * np.pi / 360
    x_threshold: float = 2.4
    x_init: float = 0.05
    sutton_barto_reward: bool = False

    screen_width: int = 600
    screen_height: int = 400


class CartPoleFunctional(
    FuncEnv[StateType, jax.Array, int, float, bool, RenderStateType, CartPoleParams]
):
    """Cartpole but in jax and functional."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def initial(
        self, rng: PRNGKeyType, params: CartPoleParams = CartPoleParams
    ) -> StateType:
        """Initial state generation."""
        return jax.random.uniform(
            key=rng, minval=-params.x_init, maxval=params.x_init, shape=(4,)
        )

    def transition(
        self,
        state: StateType,
        action: int | jax.Array,
        rng: None = None,
        params: CartPoleParams = CartPoleParams,
    ) -> StateType:
        """Cartpole transition."""
        x, x_dot, theta, theta_dot = state
        force = jnp.sign(action - 0.5) * params.force_mag
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + params.polemass_length * theta_dot**2 * sintheta
        ) / params.total_mass
        thetaacc = (params.gravity * sintheta - costheta * temp) / (
            params.length
            * (4.0 / 3.0 - params.masspole * costheta**2 / params.total_mass)
        )
        xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

        x = x + params.tau * x_dot
        x_dot = x_dot + params.tau * xacc
        theta = theta + params.tau * theta_dot
        theta_dot = theta_dot + params.tau * thetaacc

        state = jnp.array((x, x_dot, theta, theta_dot), dtype=jnp.float32)

        return state

    def observation(
        self, state: StateType, rng: Any, params: CartPoleParams = CartPoleParams
    ) -> jax.Array:
        """Cartpole observation."""
        return state

    def terminal(
        self, state: StateType, rng: Any, params: CartPoleParams = CartPoleParams
    ) -> jax.Array:
        """Checks if the state is terminal."""
        x, _, theta, _ = state

        terminated = (
            (x < -params.x_threshold)
            | (x > params.x_threshold)
            | (theta < -params.theta_threshold_radians)
            | (theta > params.theta_threshold_radians)
        )

        return terminated

    def reward(
        self,
        state: StateType,
        action: ActType,
        next_state: StateType,
        rng: Any,
        params: CartPoleParams = CartPoleParams,
    ) -> jax.Array:
        """Computes the reward for the state transition using the action."""
        x, _, theta, _ = state

        terminated = (
            (x < -params.x_threshold)
            | (x > params.x_threshold)
            | (theta < -params.theta_threshold_radians)
            | (theta > params.theta_threshold_radians)
        )

        reward = jax.lax.cond(
            params.sutton_barto_reward,
            lambda: jax.lax.cond(terminated, lambda: -1.0, lambda: 0.0),
            lambda: 1.0,
        )

        return reward

    def render_image(
        self,
        state: StateType,
        render_state: RenderStateType,
        params: CartPoleParams = CartPoleParams,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Renders an image of the state using the render state."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        screen, clock = render_state

        world_width = params.x_threshold * 2
        scale = params.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * params.length)
        cartwidth = 50.0
        cartheight = 30.0

        x = state

        surf = pygame.Surface((params.screen_width, params.screen_height))
        surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + params.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(surf, 0, params.screen_width, carty, (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        return (screen, clock), np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

    def render_init(
        self,
        params: CartPoleParams = CartPoleParams,
        screen_width: int = 600,
        screen_height: int = 400,
    ) -> RenderStateType:
        """Initialises the render state for a screen width and height."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        pygame.init()
        screen = pygame.Surface((screen_width, screen_height))
        clock = pygame.time.Clock()

        return screen, clock

    def render_close(
        self, render_state: RenderStateType, params: CartPoleParams = CartPoleParams
    ) -> None:
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        pygame.display.quit()
        pygame.quit()

    def get_default_params(self, **kwargs) -> CartPoleParams:
        """Returns the default parameters for the environment."""
        return CartPoleParams(**kwargs)


class CartPoleJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based implementation of the CartPole environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: Any):
        """Constructor for the CartPole where the kwargs are applied to the functional environment."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        env = CartPoleFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class CartPoleJaxVectorEnv(FunctionalJaxVectorEnv, EzPickle):
    """Jax-based implementation of the vectorized CartPole environment."""

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
        "jax": True,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

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

        env = CartPoleFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxVectorEnv.__init__(
            self,
            func_env=env,
            num_envs=num_envs,
            metadata=self.metadata,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
