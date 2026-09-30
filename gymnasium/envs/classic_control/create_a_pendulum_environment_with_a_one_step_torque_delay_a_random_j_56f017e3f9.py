import numpy as np
import gymnasium as gym
from gymnasium import spaces

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x):
    """Wrap angle to [-pi, pi]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class DelayedFrictionPendulumEnv(gym.Env):
    """Pendulum with one‑step torque delay, random joint friction and wind.

    Observation space: (cos(theta), sin(theta), thetadot)
    Action space: torque in [-max_torque, max_torque]
    Episode length: 200 steps (enforced internally).
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, g: float = 10.0):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.render_mode = render_mode

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # episode‑specific parameters (set in reset)
        self.friction_coeff = None  # magnitude of dry friction torque
        self.prev_action = None      # torque applied in previous step (delay)
        self.step_counter = None

    # ---------------------------------------------------------------------
    # Environment API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Random initial state – same logic as the original PendulumEnv
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            x = options.get("x_init", DEFAULT_X)
            y = options.get("y_init", DEFAULT_Y)
            high = np.array([x, y])
        low = -high
        self.state = self.np_random.uniform(low=low, high=high)

        # Episode‑specific random friction coefficient in [0.0, 0.2]
        self.friction_coeff = self.np_random.uniform(0.0, 0.2)
        # No previously applied torque at the first step (delay buffer)
        self.prev_action = 0.0
        # Step counter for internal horizon enforcement
        self.step_counter = 0
        # Rendering placeholder
        self.last_u = None
        return self._get_obs(), {}

    def step(self, action):
        # Clip incoming action to torque limits
        u = np.clip(action, -self.max_torque, self.max_torque)[0]

        # Apply the delayed torque (torque from previous step)
        applied_torque = self.prev_action

        # Stochastic wind torque (zero‑mean Gaussian noise)
        wind = self.np_random.normal(loc=0.0, scale=0.1)
        applied_torque += wind

        # Dry friction opposing the direction of angular velocity
        th, thdot = self.state
        if np.abs(thdot) > 1e-5:
            friction = -self.friction_coeff * np.sign(thdot) * self.max_torque
        else:
            # When velocity is near zero, friction can prevent motion up to a threshold
            friction = -np.clip(self.friction_coeff * self.max_torque, -np.abs(applied_torque), np.abs(applied_torque))
        applied_torque += friction

        # Dynamics (same as original PendulumEnv but using applied_torque)
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * applied_torque) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        self.last_u = u  # store the *current* (non‑delayed) action for optional rendering

        # Cost/reward – same shaping as original PendulumEnv
        costs = angle_normalize(newth) ** 2 + 0.1 * newthdot ** 2 + 0.001 * (u ** 2)
        reward = -costs

        # Update delay buffer for next step
        self.prev_action = u

        # Horizon handling
        self.step_counter += 1
        truncated = self.step_counter >= 200
        terminated = False  # pendulum never terminates early in this suite
        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    # Rendering is optional; we keep a minimal stub to satisfy the API.
    def render(self):
        if self.render_mode is None:
            return None
        # No visualisation required for this environment variant.
        return None

    def close(self):
        # No external resources to release.
        pass
