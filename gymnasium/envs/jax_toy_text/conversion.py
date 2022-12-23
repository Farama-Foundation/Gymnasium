from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import jax.random as jrng
import numpy as np

import gymnasium as gym
from gymnasium import Space
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.registration import EnvSpec
from gymnasium.experimental.functional import ActType, FuncEnv, StateType
from gymnasium.utils import seeding


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
        sutton_and_barto = True,
        natural = False
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
        self.render_state = None

        self._is_box_action_space = isinstance(self.action_space, gym.spaces.Box)

        self.func_env.sutton_and_barto = sutton_and_barto
        self.func_env.natural = natural


        np_random, _ = seeding.np_random()
        seed = np_random.integers(0, 2**32 - 1, dtype="uint32")

        self.rng = jrng.PRNGKey(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        

        
        if self.render_mode == "rgb_array" or self.render_mode == "human":
            try:
                import pygame
            except ImportError:
                raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy_text]`"
            )
            self.render_state = self.func_env.render_init(0)
        if self.render_mode == "human":
            pygame.display.flip()

        
    
        if seed is not None:
            self.rng = jrng.PRNGKey(seed)


        self.state, self.rng = self.func_env.initial(rng=self.rng)


        obs = self.func_env.observation(self.state)
        info = self.func_env.state_info(self.state)

        obs = _convert_jax_to_numpy(obs)
        obs = tuple([int(value) for value in obs])

        return obs, info

    def step(self, action: ActType):
        if self._is_box_action_space:
            assert isinstance(self.action_space, gym.spaces.Box)  # For typing
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:  # Discrete
            # For now we assume jax envs don't use complex spaces
            err_msg = f"{action!r} ({type(action)}) invalid"
            assert self.action_space.contains(action), err_msg


        next_state, self.rng = self.func_env.transition(self.state, action, self.rng)
        observation = self.func_env.observation(next_state)
        reward = self.func_env.reward(self.state, action, next_state)
        terminated = self.func_env.terminal(next_state)
        info = self.func_env.step_info(self.state, action, next_state)
        self.state = next_state

        observation = _convert_jax_to_numpy(observation)

        observation = tuple([int(value) for value in observation])

        return observation, float(reward), bool(terminated), False, info

    def render(self):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy_text]`"
            )
        if self.render_mode != None:
            self.render_state, image = self.func_env.render_image(
                self.state, self.render_state
            )
        if self.render_mode == "rgb_array":
            return image
        elif self.render_mode == "human":
            pygame.display.flip()

    def close(self):
        if self.render_state is not None:
            self.func_env.render_close(self.render_state)
            self.render_state = None


def _convert_jax_to_numpy(element: Any):
    """
    Convert a jax observation/action to a numpy array, or a numpy-based container.
    Currently required because all tests assume that stuff is in numpy arrays, hopefully will be removed soon.
    """
    if isinstance(element, jnp.ndarray):
        if element.size == 1 and len(element.shape) == 0 :
            return np.asarray(element).item()
        else:
            return np.asarray(element)
    elif isinstance(element, tuple):
        return tuple(_convert_jax_to_numpy(e) for e in element)
    elif isinstance(element, list):
        return [_convert_jax_to_numpy(e) for e in element]
    elif isinstance(element, dict):
        return {k: _convert_jax_to_numpy(v) for k, v in element.items()}
    else:
        raise TypeError(f"Cannot convert {element} to numpy")
