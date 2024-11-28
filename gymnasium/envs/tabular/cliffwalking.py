"""This module provides a CliffWalking functional environment and Gymnasium environment wrapper CliffWalkingJaxEnv."""

from __future__ import annotations

from os import path
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

from gymnasium import spaces
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import ActType, FuncEnv, StateType
from gymnasium.utils import EzPickle
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers import HumanRendering


if TYPE_CHECKING:
    import pygame


class RenderStateType(NamedTuple):
    """A named tuple which contains the full render state of the Cliffwalking Env. This is static during the episode."""

    screen: pygame.surface
    shape: tuple[int, int]
    nS: int
    cell_size: tuple[int, int]
    cliff: np.ndarray
    elf_images: tuple[pygame.Surface, pygame.Surface, pygame.Surface, pygame.Surface]
    start_img: pygame.Surface
    goal_img: pygame.Surface
    bg_imgs: tuple[str, str]
    mountain_bg_img: tuple[pygame.Surface, pygame.Surface]
    near_cliff_imgs: tuple[str, str]
    near_cliff_img: tuple[pygame.Surface, pygame.Surface]
    cliff_img: pygame.Surface


# RenderStateType =RenderState #Tuple["pygame.Surface", Tuple[int, int], int, Tuple[int, int], "numpy.ndarray", Tuple["pygame.Surface", "pygame.Surface", "pygame.Surface", "pygame.Surface"], "pygame.Surface", "pygame.Surface", Tuple[str, str], Tuple["pygame.surface", "pygame.surface"], Tuple[str, str], Tuple["pygame.surface", "pygame.surface"], "pygame.surface"]


class EnvState(NamedTuple):
    """A named tuple which contains the full state of the Cliffwalking game."""

    player_position: jnp.array
    last_action: int
    fallen: bool


def fell_off(player_position):
    """Checks to see if the player_position means the player has fallen of the cliff."""
    return (
        (player_position[0] == 3)
        * (player_position[1] >= 1)
        * (player_position[1] <= 10)
    )


class CliffWalkingFunctional(
    FuncEnv[jax.Array, jax.Array, int, float, bool, RenderStateType, None]
):
    """Cliff walking involves crossing a gridworld from start to goal while avoiding falling off a cliff.

    ## Description
    The game starts with the player at location [3, 0] of the 4x12 grid world with the
    goal located at [3, 11]. If the player reaches the goal the episode ends.

    A cliff runs along [3, 1..10]. If the player moves to a cliff location it
    returns to the start location.

    The player makes moves until they reach the goal.

    Adapted from Example 6.6 (page 132) from Reinforcement Learning: An Introduction
    by Sutton and Barto [<a href="#cliffwalk_ref">1</a>].

    With inspiration from:
    [https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py)

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move up
    - 1: Move right
    - 2: Move down
    - 3: Move left

    ## Observation Space
    There are 3 x 12 + 1 possible states. The player cannot be at the cliff, nor at
    the goal as the latter results in the end of the episode. What remains are all
    the positions of the first 3 rows plus the bottom-left cell.

    The observation is a value representing the player's current position as
    current_row * ncols + current_col (where both the row and col start at 0).

    For example, the starting position can be calculated as follows: 3 * 12 + 0 = 36.

    The observation is returned as an `numpy.ndarray` with shape `(1,)` and dtype `numpy.int32` .

    ## Starting State
    The episode starts with the player in state `[36]` (location [3, 0]).

    ## Reward
    Each time step incurs -1 reward, unless the player stepped into the cliff,
    which incurs -100 reward.

    ## Episode End
    The episode terminates when the player enters state `[47]` (location [3, 11]).


    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('tablular/CliffWalking-v0')
    ```

    ## References
    <a id="cliffwalk_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
    An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)

    ## Version History
    - v0: Initial version release

    """

    action_space = spaces.Box(low=0, high=3, dtype=np.int32)  # 4 directions
    observation_space = spaces.Box(
        low=0, high=(12 * 4) - 1, shape=(1,), dtype=np.int32
    )  # A discrete state corresponds to each possible location

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def transition(
        self,
        state: EnvState,
        action: int | jax.Array,
        key: PRNGKey,
        params: None = None,
    ):
        """The Cliffwalking environment's state transition function."""
        new_position = state.player_position

        # where is the agent trying to go?
        new_position = jnp.array(
            [
                new_position[0] + (1 * (action == 2)) + (-1 * (action == 0)),
                new_position[1] + (1 * (action == 1)) + (-1 * (action == 3)),
            ]
        )

        # prevent out of bounds
        new_position = jnp.array(
            [
                jnp.maximum(jnp.minimum(new_position[0], 3), 0),
                jnp.maximum(jnp.minimum(new_position[1], 11), 0),
            ]
        )

        # if we fell off, we have to start over from scratch from (3,0)
        fallen = fell_off(new_position)
        new_position = jnp.array(
            [
                new_position[0] * (1 - fallen) + 3 * fallen,
                new_position[1] * (1 - fallen),
            ]
        )
        new_state = EnvState(
            player_position=new_position.reshape((2,)),
            last_action=action[0],
            fallen=fallen,
        )

        return new_state

    def initial(self, rng: PRNGKey, params: None = None) -> EnvState:
        """Cliffwalking initial observation function."""
        player_position = jnp.array([3, 0])

        state = EnvState(player_position=player_position, last_action=-1, fallen=False)
        return state

    def observation(self, state: EnvState, params: None = None) -> int:
        """Cliffwalking observation."""
        return jnp.array(
            state.player_position[0] * 12 + state.player_position[1]
        ).reshape((1,))

    def terminal(self, state: EnvState, params: None = None) -> jax.Array:
        """Determines if a particular Cliffwalking observation is terminal."""
        return jnp.array_equal(state.player_position, jnp.array([3, 11]))

    def reward(
        self,
        state: EnvState,
        action: ActType,
        next_state: StateType,
        params: None = None,
    ) -> jax.Array:
        """Calculates reward from a state."""
        state = next_state
        reward = -1 + (-99 * state.fallen[0])
        return jax.lax.convert_element_type(reward, jnp.float32)

    def render_init(
        self, screen_width: int = 600, screen_height: int = 500
    ) -> RenderStateType:
        """Returns an initial render state."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            )

        cell_size = (60, 60)
        window_size = (
            4 * cell_size[0],
            12 * cell_size[1],
        )

        pygame.init()
        screen = pygame.Surface((window_size[1], window_size[0]))

        shape = (4, 12)
        nS = 4 * 12
        # Cliff Location
        cliff = np.zeros(shape, dtype=bool)
        cliff[3, 1:-1] = True

        hikers = [
            path.join(path.dirname(__file__), "../toy_text/img/elf_up.png"),
            path.join(path.dirname(__file__), "../toy_text/img/elf_right.png"),
            path.join(path.dirname(__file__), "../toy_text/img/elf_down.png"),
            path.join(path.dirname(__file__), "../toy_text/img/elf_left.png"),
        ]

        cell_size = (60, 60)

        elf_images = [
            pygame.transform.scale(pygame.image.load(f_name), cell_size)
            for f_name in hikers
        ]
        file_name = path.join(path.dirname(__file__), "../toy_text/img/stool.png")
        start_img = pygame.transform.scale(pygame.image.load(file_name), cell_size)
        file_name = path.join(path.dirname(__file__), "../toy_text/img/cookie.png")
        goal_img = pygame.transform.scale(pygame.image.load(file_name), cell_size)
        bg_imgs = [
            path.join(path.dirname(__file__), "../toy_text/img/mountain_bg1.png"),
            path.join(path.dirname(__file__), "../toy_text/img/mountain_bg2.png"),
        ]
        mountain_bg_img = [
            pygame.transform.scale(pygame.image.load(f_name), cell_size)
            for f_name in bg_imgs
        ]
        near_cliff_imgs = [
            path.join(
                path.dirname(__file__), "../toy_text/img/mountain_near-cliff1.png"
            ),
            path.join(
                path.dirname(__file__), "../toy_text/img/mountain_near-cliff2.png"
            ),
        ]
        near_cliff_img = [
            pygame.transform.scale(pygame.image.load(f_name), cell_size)
            for f_name in near_cliff_imgs
        ]
        file_name = path.join(
            path.dirname(__file__), "../toy_text/img/mountain_cliff.png"
        )
        cliff_img = pygame.transform.scale(pygame.image.load(file_name), cell_size)

        return RenderStateType(
            screen=screen,
            shape=shape,
            nS=nS,
            cell_size=cell_size,
            cliff=cliff,
            elf_images=tuple(elf_images),
            start_img=start_img,
            goal_img=goal_img,
            bg_imgs=tuple(bg_imgs),
            mountain_bg_img=tuple(mountain_bg_img),
            near_cliff_imgs=tuple(near_cliff_imgs),
            near_cliff_img=tuple(near_cliff_img),
            cliff_img=cliff_img,
        )

    def render_image(
        self, state: StateType, render_state: RenderStateType, params: None = None
    ) -> tuple[RenderStateType, np.ndarray]:
        """Renders an image from a state."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy_text]"`'
            )
        (
            window_surface,
            shape,
            nS,
            cell_size,
            cliff,
            elf_images,
            start_img,
            goal_img,
            bg_imgs,
            mountain_bg_img,
            near_cliff_imgs,
            near_cliff_img,
            cliff_img,
        ) = render_state

        for s in range(nS):
            row, col = np.unravel_index(s, shape)
            pos = (col * cell_size[0], row * cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            window_surface.blit(mountain_bg_img[check_board_mask], pos)

            if cliff[row, col]:
                window_surface.blit(cliff_img, pos)
            if row < shape[0] - 1 and cliff[row + 1, col]:
                window_surface.blit(near_cliff_img[check_board_mask], pos)
            if s == 36:
                window_surface.blit(start_img, pos)
            if s == nS - 1:
                window_surface.blit(goal_img, pos)
            if s == state.player_position[0] * 12 + state.player_position[1]:
                elf_pos = (pos[0], pos[1] - 0.1 * cell_size[1])
                last_action = state.last_action if state.last_action != -1 else 2
                window_surface.blit(elf_images[last_action], elf_pos)

        return render_state, np.transpose(
            np.array(pygame.surfarray.pixels3d(window_surface)), axes=(1, 0, 2)
        )

    def render_close(self, render_state: RenderStateType) -> None:
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e
        pygame.display.quit()
        pygame.quit()


class CliffWalkingJaxEnv(FunctionalJaxEnv, EzPickle):
    """A Gymnasium Env wrapper for the functional cliffwalking env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs):
        """Initializes Gym wrapper for cliffwalking functional env."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)
        env = CliffWalkingFunctional(**kwargs)
        env.transform(jax.jit)

        super().__init__(
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


if __name__ == "__main__":
    """
    Temporary environment tester function.
    """

    env = HumanRendering(CliffWalkingJaxEnv(render_mode="rgb_array"))

    obs, info = env.reset()
    print(obs, info)

    terminal = False
    while not terminal:
        action = int(input("Please input an action\n"))
        obs, reward, terminal, truncated, info = env.step(action)
        print(obs, reward, terminal, truncated, info)

    exit()
