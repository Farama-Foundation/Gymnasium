import os
import json
import ast
import hashlib
import re
import math

# Generated environments are written as importable Gymnasium modules.
output_dir = os.path.join("gymnasium", "envs", "classic_control")
os.makedirs(output_dir, exist_ok=True)

# System prompt from LLM.py
system_prompt = """
You are an expert in python programming and Reinforcement Learning. Your goal is to provide the next task for an agent looking to learn a collection of tasks in an open-ended fashion. You will be provided with a list of tasks and how well the agent does well there as compared to a random agent. Your task is to analyze the current level of the agent and write code for the next environment the agent should learn via RL. You are only allowed to change the reward functions and the initial configurations, not the naturer of the robot/agent(action/observation space).
The suggested task must be:
1) Learnable: not too difficult for the agent based on its current level.
2) Feasible: implementable as a self-contained Gymnasium environment.
3) Novel: not already present in the existing environment list.
4) Interesting: worth learning according to human notions of interestingness.
5) Diverse: vary dynamics, rewards, observations, actions, or initial conditions.

Return a complete Python module for one new environment.
The module must import numpy and gymnasium, define a class inheriting from
gymnasium.Env, keep the action_space and observation_space the same for all tasks, and implement
reset(seed=None, options=None), step(action), render(), and close() with changes making the tasks  more interesting/diffiult. Follow
the Gymnasium API: reset returns (observation, info), and step returns
(observation, reward, terminated, truncated, info). Use only dependencies
available in this repository plus numpy. Include a module-level metadata
dictionary and a clear class name ending in Env.
"""

# Example code from c.py to provide as reference

initial_environment = """
from os import path

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class PendulumEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: str | None = None, g=10.0):
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
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the gymnasium api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

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
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
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
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
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
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


"""

# Track previously generated and learned environment concepts.
def generate_environment(learned_titles, performance_report, output_dir=None):
    """Generate and save one environment from measured agent performance."""
    from groq import Groq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY must be set before generating an environment")

    output_dir = output_dir or os.path.join("gymnasium", "envs", "classic_control")
    os.makedirs(output_dir, exist_ok=True)
    user_prompt = f"""
Here is the base environment from which learning started
reference:
```python
{initial_environment}
```

The agents have currently successfully learned the following environments:
{learned_titles if learned_titles else "None so far."}

Measured random-agent versus learned-agent performance:
{json.dumps(performance_report, indent=2)}

The next environment must keep the exact Pendulum observation and action spaces,
because it will be trained by the same DDPG implementation.

Please reason briefly about what RL environment the agents should learn next.
Output one valid JSON object with string fields 'reasoning', 'task', and 'code'.
The code field must contain one complete Python source module, with a gymnasium.Env
class implementing reset, step, render, and close. Do not truncate the module,
omit methods, or wrap the JSON in Markdown fences. The environment must enforce a
200-step episode horizon: reset must set an episode step counter to zero, step must
increment it, and return truncated=True when the counter reaches 200 unless the
episode has already terminated. Keep the truncation logic inside the environment
as a defensive fallback; the runner also applies Gymnasium's TimeLimit wrapper.
"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_completion_tokens=12000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content or "{}")
    title = result.get("task", "generated_environment")
    code_to_run = result.get("code", "")
    code_to_run = re.sub(r"^```(?:python)?\s*|\s*```$", "", code_to_run.strip())
    if not code_to_run:
        raise RuntimeError("Groq returned no environment code")
    module_tree = ast.parse(code_to_run)
    environment_classes = [
        node
        for node in module_tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Attribute) and base.attr == "Env"
            for base in node.bases
        )
    ]
    required_methods = {"reset", "step", "render", "close"}
    if not environment_classes:
        raise RuntimeError("Groq returned code without a gymnasium.Env class")
    method_names = {
        node.name
        for node in environment_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing_methods = required_methods - method_names
    if missing_methods:
        raise RuntimeError(
            "Groq returned an incomplete environment; missing methods: "
            + ", ".join(sorted(missing_methods))
        )

    module_name = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    module_name = module_name or "generated_environment"
    if len(module_name) > 80:
        title_hash = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
        module_name = f"{module_name[:69].rstrip('_')}_{title_hash}"
    module_path = os.path.join(output_dir, f"{module_name}.py")
    with open(module_path, "w", encoding="utf-8") as environment_file:
        environment_file.write(code_to_run.rstrip() + "\n")

    return {
        "title": title,
        "reasoning": result.get("reasoning", ""),
        "module_name": module_name,
        "module_path": module_path,
    }
