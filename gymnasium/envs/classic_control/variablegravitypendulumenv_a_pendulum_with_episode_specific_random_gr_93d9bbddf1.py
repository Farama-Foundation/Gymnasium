import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x: float) -> float:
    """Normalize angle to [-π, π]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class VariableGravityPendulumEnv(gym.Env):
    """Pendulum with a random gravity value each episode.

    The observation space, action space and rendering are identical to the
    classic ``Pendulum-v1`` environment.  The only differences are:
    * ``g`` (gravity) is sampled uniformly from ``[5.0, 15.0]`` at reset.
    * Episodes are limited to 200 timesteps via an internal counter.
    * The reward penalises deviation from the upright position (θ=0) and
      torque usage, mirroring the original cost formulation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        # dynamics parameters (same as original pendulum)
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.m = 1.0
        self.l = 1.0
        # episode management
        self.max_steps = 200
        self.current_step = 0
        # rendering
        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        # spaces (identical to Pendulum-v1)
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        # placeholder for gravity – will be set in reset()
        self.g = 10.0
        self.state = None
        self.last_u = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Sample a new gravity for the episode
        self.g = float(self.np_random.uniform(5.0, 15.0))
        # Initialise the pendulum state – keep the same API as the original env
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
        self.current_step = 0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        th, thdot = self.state
        # Clip and extract torque
        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        # Cost/reward – identical to original but with current gravity
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        reward = -costs
        # Dynamics (same equations as original Pendulum)
        newthdot = thdot + (
            3.0 * self.g / (2.0 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * u
        ) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        self.state = np.array([newth, newthdot])
        # Episode bookkeeping
        self.current_step += 1
        terminated = False  # no explicit terminal condition
        truncated = self.current_step >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    # ---------------------------------------------------------------------
    # Rendering – copied verbatim from the reference Pendulum implementation
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
        # Background
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))
        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2
        # Rod geometry
        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed = []
        for c in coords:
            vec = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            transformed.append((vec[0] + offset, vec[1] + offset))
        gfxdraw.aapolygon(self.surf, transformed, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed, (204, 77, 77))
        # Axle
        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        # End of rod
        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        # Torque visualisation (clockwise.png asset expected next to this file)
        fname = utils.path_join(utils.path_dirname(__file__), "assets/clockwise.png")
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
                flip = bool(self.last_u > 0)
                scale_img = pygame.transform.flip(scale_img, flip, True)
                self.surf.blit(
                    scale_img,
                    (offset - scale_img.get_rect().centerx, offset - scale_img.get_rect().centery),
                )
        except Exception:
            # If the asset is missing we simply skip the torque overlay.
            pass
        # Center axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        # Flip for correct orientation
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
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
