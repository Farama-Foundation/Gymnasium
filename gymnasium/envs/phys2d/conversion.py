from typing import Any, Dict, Optional, Tuple

import jax.random as jrng

import gymnasium as gym
from gymnasium import Space
from gymnasium.envs.registration import EnvSpec
from gymnasium.functional import ActType, FuncEnv, StateType


class JaxEnv(gym.Env):
    """
    A conversion layer for numpy-based environments.
    """

    state: StateType
    rng: jrng.PRNGKey

    def __init__(
        self,
        func_env: FuncEnv,
        observation_space: Space,
        action_space: Space,
        metadata: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        reward_range: Tuple[float, float] = (-float("inf"), float("inf")),
        spec: Optional[EnvSpec] = None,
    ):
        """Initialize the environment from a FuncEnv."""
        if metadata is None:
            metadata = {}
        self.func_env = func_env
        self.observation_space = observation_space
        self.action_space = action_space
        self.metadata = metadata
        self.render_mode = render_mode
        self.reward_range = reward_range
        self.spec = spec

        if self.render_mode == "rgb_array":
            self.render_state = self.func_env.render_init()
        else:
            self.render_state = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if not hasattr(self, "rng") or seed is not None:
            self.rng = jrng.PRNGKey(0 if seed is None else seed)

        rng, self.rng = jrng.split(self.rng)

        self.state = self.func_env.initial(rng=rng)
        obs = self.func_env.observation(self.state)
        info = self.func_env.state_info(self.state)

        return obs, info

    def step(self, action: ActType):
        rng, self.rng = jrng.split(self.rng)

        next_state = self.func_env.transition(self.state, action, rng)
        observation = self.func_env.observation(self.state)
        reward = self.func_env.reward(self.state, action, next_state)
        terminated = self.func_env.terminal(next_state)
        info = self.func_env.step_info(self.state, action, next_state)
        self.state = next_state

        return observation, float(reward), bool(terminated), False, info

    def render(self):
        if self.render_mode == "rgb_array":
            self.render_state, image = self.func_env.render_image(
                self.state, self.render_state
            )
            return image
        else:
            raise NotImplementedError
