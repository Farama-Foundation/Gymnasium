import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class VariableGravityWindPendulumEnv(gym.Env):
    """Pendulum with per‑step sinusoidal gravity and stochastic wind torque.

    The observation space and action space are identical to the classic
    ``Pendulum-v1`` environment:
        observation = [cos(theta), sin(theta), thetadot]
        action = torque in [-max_torque, max_torque]

    At the start of each episode a random gravity amplitude and frequency are
    sampled. Gravity at step *t* is:
        g_t = g_base + amp * sin(2π * freq * t / max_steps)
    Additionally a zero‑mean Gaussian wind torque is added to the agent's
    command each step.

    The episode is terminated after ``max_steps`` (=200) steps by setting the
    ``truncated`` flag. No other termination condition is used.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: str | None = None, max_steps: int = 200):
        super().__init__()
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.base_g = 10.0  # nominal gravity
        self.m = 1.0
        self.l = 1.0

        self.max_steps = max_steps
        self.render_mode = render_mode

        # Rendering helpers (same as original PendulumEnv)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------
    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Random initial angle and angular velocity (same logic as original)
        if options is None:
            high = np.array([np.pi, 1.0])
        else:
            x = options.get("x_init", np.pi)
            y = options.get("y_init", 1.0)
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        # Episode‑specific sinusoid parameters for gravity
        self.gravity_amp = self.np_random.uniform(0.0, 5.0)  # amplitude up to 5
        self.gravity_freq = self.np_random.uniform(0.5, 3.0)  # cycles per episode
        self.wind_std = 0.2  # standard deviation of wind torque
        self.step_counter = 0

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        th, thdot = self.state
        # Clip agent torque
        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        # Sample wind torque and add
        wind = self.np_random.normal(0.0, self.wind_std)
        total_torque = u + wind
        self.last_u = total_torque

        # Compute current gravity for this step
        g_t = self.base_g + self.gravity_amp * np.sin(
            2 * np.pi * self.gravity_freq * self.step_counter / self.max_steps
        )

        # Dynamics (same equations as original PendulumEnv, using g_t)
        costs = (
            self.angle_normalize(th) ** 2
            + 0.1 * thdot ** 2
            + 0.001 * (total_torque ** 2)
        )
        newthdot = thdot + (
            3 * g_t / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * total_torque
        ) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        self.state = np.array([newth, newthdot])

        # Increment step counter and handle horizon truncation
        self.step_counter += 1
        truncated = self.step_counter >= self.max_steps
        terminated = False  # No internal termination condition

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, terminated, truncated, {}

    # ---------------------------------------------------------------------
    # Rendering (copy of original PendulumEnv rendering logic)
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
            else:
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
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))

        # Visualise the total torque (agent + wind) with an arrow image
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
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )
        except Exception:
            # If the image cannot be loaded (e.g., missing asset), ignore.
            pass

        # Axle
        gfxdraw.aacircle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))

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
