"""Implementation of a Jax-accelerated cartpole environment."""
from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.func_jax_env import FunctionalJaxEnv
from gymnasium.experimental.functional import ActType, FuncEnv, StateType
from gymnasium.utils import EzPickle


RenderStateType = Tuple["pygame.Surface", "pygame.time.Clock"]  # type: ignore  # noqa: F821


class CartPoleFunctional(
    FuncEnv[jnp.ndarray, jnp.ndarray, int, float, bool, RenderStateType]
):
    """Cartpole but in jax and functional.

    Example usage:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from gymnasium.envs.phys2d.cartpole import CartPoleFunctional

        >>> key = jax.random.PRNGKey(0)

        >>> env = CartPoleFunctional({"x_init": 0.5})
        >>> state = env.initial(key)
        >>> print(state)
        [ 0.46532142 -0.27484107  0.13302994 -0.20361817]
        >>> print(env.transition(state, 0))
        [ 0.4598246  -0.6357784   0.12895757  0.1278053 ]

        >>> env.transform(jax.jit)

        >>> state = env.initial(key)
        >>> print(state)
        [ 0.46532142 -0.27484107  0.13302994 -0.20361817]
        >>> print(env.transition(state, 0))
        [ 0.4598246  -0.6357784   0.12895757  0.12780523]

        >>> vkey = jax.random.split(key, 10)
        >>> env.transform(jax.vmap)
        >>> vstate = env.initial(vkey)
        >>> print(vstate)
        [[ 0.25117755 -0.03159595  0.09428263  0.12404168]
         [ 0.231457    0.41420317 -0.13484478  0.29151905]
         [-0.11706758 -0.37130308  0.13587534  0.33141208]
         [-0.4613737   0.36557996  0.3950702   0.3639989 ]
         [-0.14707637 -0.34273267 -0.32374108 -0.48110402]
         [-0.45774353  0.3633288  -0.3157575  -0.03586268]
         [ 0.37344885 -0.279778   -0.33894253  0.07415426]
         [-0.20234215  0.39775252 -0.2556088   0.32877135]
         [-0.2572986  -0.29943776 -0.45600426 -0.35740316]
         [ 0.05436695  0.35021234 -0.36484408  0.2805779 ]]
        >>> print(env.transition(vstate, jnp.array([0 for _ in range(10)])))
        [[ 0.25054562 -0.38763174  0.09676346  0.4448946 ]
         [ 0.23974106  0.09849604 -0.1290144   0.5390002 ]
         [-0.12449364 -0.7323911   0.14250359  0.6634313 ]
         [-0.45406207 -0.01028753  0.4023502   0.7505522 ]
         [-0.15393102 -0.6168968  -0.33336315 -0.30407968]
         [-0.45047694  0.08870795 -0.31647477  0.14311607]
         [ 0.36785328 -0.54895645 -0.33745944  0.24393772]
         [-0.19438711  0.10855066 -0.24903338  0.5316877 ]
         [-0.26328734 -0.5420943  -0.46315232 -0.2344252 ]
         [ 0.06137119  0.08665388 -0.35923252  0.4403924 ]]
    """

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole + length
    force_mag = 10.0
    tau = 0.02
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4
    x_init = 0.05

    screen_width = 600
    screen_height = 400

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def initial(self, rng: PRNGKey):
        """Initial state generation."""
        return jax.random.uniform(
            key=rng, minval=-self.x_init, maxval=self.x_init, shape=(4,)
        )

    def transition(
        self, state: jnp.ndarray, action: int | jnp.ndarray, rng: None = None
    ) -> StateType:
        """Cartpole transition."""
        x, x_dot, theta, theta_dot = state
        force = jnp.sign(action - 0.5) * self.force_mag
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        state = jnp.array((x, x_dot, theta, theta_dot), dtype=jnp.float32)

        return state

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        """Cartpole observation."""
        return state

    def terminal(self, state: jnp.ndarray) -> jnp.ndarray:
        """Checks if the state is terminal."""
        x, _, theta, _ = state

        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        return terminated

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> jnp.ndarray:
        """Computes the reward for the state transition using the action."""
        x, _, theta, _ = state

        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        reward = jax.lax.cond(terminated, lambda: 0.0, lambda: 1.0)
        return reward

    def render_image(
        self,
        state: StateType,
        render_state: RenderStateType,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Renders an image of the state using the render state."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e
        screen, clock = render_state

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        x = state

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
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

        gfxdraw.hline(surf, 0, self.screen_width, carty, (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        return (screen, clock), np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

    def render_init(
        self, screen_width: int = 600, screen_height: int = 400
    ) -> RenderStateType:
        """Initialises the render state for a screen width and height."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        pygame.init()
        screen = pygame.Surface((screen_width, screen_height))
        clock = pygame.time.Clock()

        return screen, clock

    def render_close(self, render_state: RenderStateType) -> None:
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e
        pygame.display.quit()
        pygame.quit()


class CartPoleJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based implementation of the CartPole environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, **kwargs: Any):
        """Constructor for the CartPole where the kwargs are applied to the functional environment."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        env = CartPoleFunctional(**kwargs)
        env.transform(jax.jit)

        action_space = env.action_space
        observation_space = env.observation_space

        super().__init__(
            env,
            observation_space=observation_space,
            action_space=action_space,
            metadata=self.metadata,
            render_mode=render_mode,
        )
