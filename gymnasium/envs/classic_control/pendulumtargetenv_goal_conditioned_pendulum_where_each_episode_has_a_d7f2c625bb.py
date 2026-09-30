import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.classic_control import utils

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x):
    """Wrap angle to [-π, π]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumTargetEnv(gym.Env):
    """Goal‑conditioned pendulum.

    The agent receives the same observation as the classic Pendulum environment
    (cos(theta), sin(theta), theta_dot) and the same torque action space.  At the
    beginning of each episode a target angle ``target`` is sampled uniformly from
    ``[-π, π]`` (or can be provided via the ``options`` dict).  The reward is
    ``-(angle_error**2 + 0.1*theta_dot**2 + 0.001*torque**2)`` where
    ``angle_error = angle_normalize(theta - target)``.  The episode ends only
    via the external ``TimeLimit`` wrapper.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, g: float = 10.0):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.render_mode = render_mode

        # identical spaces to Pendulum-v1
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # rendering helpers (copied from original Pendulum)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state = None
        self.target = None
        self.last_u = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # initial pendulum state
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            x = options.get("x_init", DEFAULT_X)
            y = options.get("y_init", DEFAULT_Y)
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        # target angle
        if options is not None and "target_angle" in options:
            self.target = float(options["target_angle"])
        else:
            self.target = self.np_random.uniform(-np.pi, np.pi)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {"target": self.target}

    def step(self, u):
        th, thdot = self.state
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u

        # dynamics (identical to classic Pendulum)
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        self.state = np.array([newth, newthdot])

        # reward based on distance to target angle
        angle_error = angle_normalize(newth - self.target)
        costs = angle_error ** 2 + 0.1 * newthdot ** 2 + 0.001 * (u ** 2)
        reward = -costs

        if self.render_mode == "human":
            self.render()
        # No internal termination; external TimeLimit will truncate.
        return self._get_obs(), reward, False, False, {"target": self.target}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    # Rendering code is largely copied from the original Pendulum environment,
    # with an additional small marker indicating the target angle.
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, e.g. gym.make(\"PendulumTarget-v0\", render_mode=\"rgb_array\")"
            )
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.display.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.screen_dim, self.screen_dim))
        surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        # Draw pendulum rod
        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed = []
        for c in coords:
            vec = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            transformed.append((vec[0] + offset, vec[1] + offset))
        gfxdraw.aapolygon(surf, transformed, (204, 77, 77))
        gfxdraw.filled_polygon(surf, transformed, (204, 77, 77))

        # axle
        gfxdraw.aacircle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        # draw end of rod
        end = pygame.math.Vector2(rod_length, 0).rotate_rad(self.state[0] + np.pi / 2)
        end_pos = (int(end[0] + offset), int(end[1] + offset))
        gfxdraw.aacircle(surf, end_pos[0], end_pos[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(surf, end_pos[0], end_pos[1], int(rod_width / 2), (204, 77, 77))

        # torque visualisation (same as original)
        fname = ""  # not loading external images to keep the module self‑contained
        if self.last_u is not None:
            # draw a simple line proportional to torque
            line_len = scale * np.abs(self.last_u) / 2
            angle = self.state[0] + (np.pi if self.last_u > 0 else 0)
            tip = pygame.math.Vector2(line_len, 0).rotate_rad(angle)
            tip_pos = (int(offset + tip[0]), int(offset + tip[1]))
            color = (0, 0, 255) if self.last_u > 0 else (255, 0, 0)
            gfxdraw.line(surf, offset, offset, tip_pos[0], tip_pos[1], color)

        # target marker – a small green line at the target angle
        target_len = rod_length * 0.9
        target_vec = pygame.math.Vector2(target_len, 0).rotate_rad(self.target + np.pi / 2)
        target_pos = (int(offset + target_vec[0]), int(offset + target_vec[1]))
        gfxdraw.line(surf, offset, offset, target_pos[0], target_pos[1], (0, 200, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
