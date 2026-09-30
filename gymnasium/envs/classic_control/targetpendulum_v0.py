import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class TargetPendulumEnv(gym.Env):
    """Pendulum with a random target angle and varying gravity.

    Observation space: same as the classic Pendulum (cos(theta), sin(theta), theta_dot).
    Action space: torque in [-max_torque, max_torque].
    Reward: - (angle_error**2 + 0.1*theta_dot**2 + 0.001*torque**2).
    The target angle is sampled uniformly from [-pi, pi] at reset and stays fixed for the episode.
    Gravity ``g`` is also sampled uniformly from [5, 15] each episode to diversify dynamics.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.m = 1.0
        self.l = 1.0
        self.render_mode = render_mode

        # dynamics parameters that will change per episode
        self.g = 10.0  # placeholder, overwritten in reset
        self.target_angle = 0.0

        # rendering helpers
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # ---------------------------------------------------------------------
    # Core Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Randomize gravity for diversity
        self.g = self.np_random.uniform(5.0, 15.0)
        # Sample a new target angle
        self.target_angle = self.np_random.uniform(-np.pi, np.pi)
        # Initialise state (theta, theta_dot)
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
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, u):
        th, thdot = self.state
        # Clip torque
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        # Physics update (same as classic pendulum)
        newthdot = thdot + (3.0 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        self.state = np.array([newth, newthdot])
        # Reward based on distance to target angle
        angle_error = angle_normalize(newth - self.target_angle)
        reward = -(angle_error ** 2 + 0.1 * newthdot ** 2 + 0.001 * (u ** 2))
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), False, False, {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    # ---------------------------------------------------------------------
    # Rendering (adapted from the original Pendulum env)
    # ---------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, e.g. gym.make(\"TargetPendulum-v0\", render_mode=\"rgb_array\")"
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
        # Clear background
        surf = pygame.Surface((self.screen_dim, self.screen_dim))
        surf.fill((255, 255, 255))
        # Pendulum geometry
        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2
        rod_length = scale * 1.0
        rod_width = scale * 0.2
        # Draw rod
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed = []
        for c in coords:
            vec = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            transformed.append((vec[0] + offset, vec[1] + offset))
        gfxdraw.aapolygon(surf, transformed, (204, 77, 77))
        gfxdraw.filled_polygon(surf, transformed, (204, 77, 77))
        # Draw axle
        gfxdraw.aacircle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        # Draw rod end
        end = pygame.math.Vector2(rod_length, 0).rotate_rad(self.state[0] + np.pi / 2)
        end_pos = (int(end[0] + offset), int(end[1] + offset))
        gfxdraw.aacircle(surf, end_pos[0], end_pos[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(surf, end_pos[0], end_pos[1], int(rod_width / 2), (204, 77, 77))
        # Draw target angle indicator (small red dot on the circle)
        target_vec = pygame.math.Vector2(rod_length, 0).rotate_rad(self.target_angle + np.pi / 2)
        target_pos = (int(target_vec[0] + offset), int(target_vec[1] + offset))
        gfxdraw.aacircle(surf, target_pos[0], target_pos[1], int(0.07 * scale), (255, 0, 0))
        gfxdraw.filled_circle(surf, target_pos[0], target_pos[1], int(0.07 * scale), (255, 0, 0))
        # Flip for correct orientation
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
