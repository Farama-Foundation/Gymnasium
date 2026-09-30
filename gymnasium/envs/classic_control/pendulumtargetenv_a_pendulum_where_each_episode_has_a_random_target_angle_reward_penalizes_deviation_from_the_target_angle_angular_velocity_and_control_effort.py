import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x):
    """Normalize angle to [-pi, pi]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumTargetEnv(gym.Env):
    """Pendulum environment with a random target angle each episode.

    Observation space: same as classic Pendulum – [cos(theta), sin(theta), theta_dot]
    Action space: torque in [-max_torque, max_torque]
    Reward: -(angle_error^2 + 0.1*theta_dot^2 + 0.001*torque^2) where angle_error is the
            difference between the current angle and the episode's target angle.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: str | None = None, g: float = 10.0):
        super().__init__()
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # target angle will be set in reset()
        self.target_angle = 0.0

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Initialise pendulum state
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
        # Choose a target angle for this episode – uniform in [-pi, pi]
        self.target_angle = self.np_random.uniform(low=-np.pi, high=np.pi)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {"target_angle": float(self.target_angle)}

    def step(self, u):
        th, thdot = self.state  # current angle and angular velocity
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u

        # Reward shaping – distance to target angle
        angle_error = angle_normalize(th - self.target_angle)
        costs = angle_error ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        # Physics update (same as classic pendulum)
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # No explicit termination – rely on TimeLimit wrapper
        return self._get_obs(), -costs, False, False, {"target_angle": float(self.target_angle)}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    # ---------------------------------------------------------------------
    # Rendering (copied from classic Pendulum for visual consistency)
    # ---------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
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
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.screen_dim, self.screen_dim))
        surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

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

        # torque visual cue (same as original)
        if self.last_u is not None:
            fname = "assets/clockwise.png"
            try:
                img = pygame.image.load(fname)
                scale_img = pygame.transform.smoothscale(
                    img,
                    (
                        float(scale * np.abs(self.last_u) / 2),
                        float(scale * np.abs(self.last_u) / 2),
                    ),
                )
                flip = bool(self.last_u > 0)
                scale_img = pygame.transform.flip(scale_img, flip, True)
                surf.blit(
                    scale_img,
                    (
                        offset - scale_img.get_rect().centerx,
                        offset - scale_img.get_rect().centery,
                    ),
                )
            except Exception:
                # If the asset is missing we simply ignore the visual cue.
                pass

        # flip for correct orientation
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
