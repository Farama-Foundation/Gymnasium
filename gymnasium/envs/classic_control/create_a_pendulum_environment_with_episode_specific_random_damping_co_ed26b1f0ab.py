import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class DampedWindPendulumEnv(gym.Env):
    """Pendulum with linear damping and stochastic wind torque.

    The observation space is identical to the classic PendulumEnv:
        observation = [cos(theta), sin(theta), theta_dot]
    The action space is a single torque value in [-max_torque, max_torque].

    Each episode samples a random damping coefficient, mass, length and wind
    torque standard deviation.  The dynamics are:
        theta_ddot = (3*g/(2*l))*sin(theta) + 3/(m*l**2)*u 
                     - damping*theta_dot + wind_torque
    where wind_torque ~ Normal(0, wind_std) and u is the clipped action.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, g: float = 10.0, max_steps: int = 200):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.render_mode = render_mode
        self.max_steps = max_steps

        # episode‑specific parameters (filled in reset)
        self.m = 1.0          # mass
        self.l = 1.0          # length
        self.damping = 0.0    # linear damping coefficient
        self.wind_std = 0.0   # std of wind torque noise
        self.step_counter = 0

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # rendering attributes (copied from PendulumEnv)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

    # ---------------------------------------------------------------------
    # Core Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Sample episode‑specific dynamics
        self.m = self.np_random.uniform(0.5, 2.0)   # mass between 0.5 and 2.0 kg
        self.l = self.np_random.uniform(0.5, 2.0)   # length between 0.5 and 2.0 m
        self.damping = self.np_random.uniform(0.0, 0.2)   # damping coefficient
        self.wind_std = self.np_random.uniform(0.0, 1.0)  # wind torque noise std
        self.step_counter = 0

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

    def step(self, action):
        th, thdot = self.state
        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        self.last_u = u

        # stochastic wind torque (zero‑mean Gaussian)
        wind = self.np_random.normal(0.0, self.wind_std)

        # dynamics with damping and wind
        newthdot = (
            thdot
            + (
                3 * self.g / (2 * self.l) * np.sin(th)
                + 3.0 / (self.m * self.l ** 2) * u
                - self.damping * thdot
                + wind
            )
            * self.dt
        )
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt

        self.state = np.array([newth, newthdot])
        self.step_counter += 1

        # Reward: same shaping as original Pendulum but with extra penalty for wind energy
        costs = angle_normalize(newth) ** 2 + 0.1 * newthdot ** 2 + 0.001 * (u ** 2) + 0.05 * (wind ** 2)
        reward = -costs

        terminated = False  # pendulum never terminates early in this design
        truncated = self.step_counter >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, e.g. gym.make(\"DampedWindPendulumEnv\", render_mode=\"rgb_array\")"
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

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))

        # visualize the applied torque as in the original env
        fname = utils.path.join(utils.path.dirname(__file__), "assets/clockwise.png")
        try:
            img = pygame.image.load(fname)
            if self.last_u is not None:
                scale_img = pygame.transform.smoothscale(
                    img,
                    (
                        float(scale * np.abs(self.last_u) / 2),
                        float(scale * np.abs(self.last_u) / 2),
                    ),
                )
                is_flip = bool(self.last_u > 0)
                scale_img = pygame.transform.flip(scale_img, is_flip, True)
                self.surf.blit(
                    scale_img,
                    (offset - scale_img.get_rect().centerx, offset - scale_img.get_rect().centery),
                )
        except Exception:
            # If asset missing, ignore visual torque indicator
            pass

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
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

    # ---------------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------------
    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
