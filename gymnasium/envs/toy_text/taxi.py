import itertools
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from enum import IntEnum
from io import StringIO
from os import path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled

if TYPE_CHECKING:
    import pygame

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]
WINDOW_SIZE = (550, 350)

# MAP character constants
WALL_VERTICAL = b"|"
WALL_HORIZONTAL = b"-"
PASSABLE = b":"


@dataclass
class TaxiState:
    """Represents the complete state of a Taxi environment episode."""

    s: Annotated[
        int,
        """Encoded representation of the current locations of the taxi and the
        passenger as well as the passenger's current destination.""",
    ]
    lastaction: Annotated[
        int | None,
        """The last action taken in the episode.
        This is None at the start of the episode.""",
    ]
    fickle_step: Annotated[
        bool,
        """Whether the passenger will change destinations after the taxi
        moves one step with the passenger inside.""",
    ]
    taxi_orientation: Annotated[
        str,
        """The image to use for the taxi facing.
        This is one of 'taxi_0', 'taxi_1', 'taxi_2', 'taxi_3'
        Used for rendering.""",
    ]
    np_random_state: Annotated[Mapping[str, Any], """The numpy random number state."""]


class Locations(IntEnum):
    """Possible locations for the passenger."""

    RED = 0
    GREEN = 1
    YELLOW = 2
    BLUE = 3
    TAXI = 4


class Actions(IntEnum):
    """Possible actions the agent can take."""

    MOVE_SOUTH = 0
    MOVE_NORTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    PICKUP = 4
    DROPOFF = 5


class TaxiEnv(Env):
    """
    The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them
    off at one of four locations.

    ## Description
    There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the
    5x5 grid world. The taxi starts off at a random square and the passenger at one of the
    designated locations.

    The goal is move the taxi to the passenger's location, pick up the passenger,
    move to the passenger's desired destination, and
    drop off the passenger. Once the passenger is dropped off, the episode ends.

    The player receives positive rewards for successfully dropping-off the passenger at the correct
    location. Negative rewards for incorrect attempts to pick-up/drop-off passenger and
    for each step where another reward is not received.

    Map:

            +---------+
            |R: | : :G|
            | : | : : |
            | : : : : |
            | | : | : |
            |Y| : |B: |
            +---------+

    From "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich [<a href="#taxi_ref">1</a>].

    ## Action Space
    The action shape is `(1,)` in the range `{0, 5}` indicating
    which direction to move the taxi or to pickup/drop off passengers.

    - 0: Move south (down)
    - 1: Move north (up)
    - 2: Move east (right)
    - 3: Move west (left)
    - 4: Pickup passenger
    - 5: Drop off passenger

    ## Observation Space
    There are 500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), and 4 destination locations.

    Destination on the map are represented with the first letter of the color.

    Passenger locations:
    - 0: Red
    - 1: Green
    - 2: Yellow
    - 3: Blue
    - 4: In taxi

    Destinations:
    - 0: Red
    - 1: Green
    - 2: Yellow
    - 3: Blue

    An observation is returned as an `int()` that encodes the corresponding state, calculated by
    `((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination`

    Note that there are 400 states that can actually be reached during an
    episode. The missing states correspond to situations in which the passenger
    is at the same location as their destination, as this typically signals the
    end of an episode. Four additional states can be observed right after a
    successful episodes, when both the passenger and the taxi are at the destination.
    This gives a total of 404 reachable discrete states.

    ## Starting State
    The initial state is sampled uniformly from the possible states
    where the passenger is neither at their destination nor inside the taxi.
    There are 300 possible initial states: 25 taxi positions, 4 passenger locations (excluding inside the taxi)
    and 3 destinations (excluding the passenger's current location).

    ## Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10  executing "pickup" and "drop-off" actions illegally.

    An action that results a noop, like moving into a wall, will incur the time step
    penalty. Noops can be avoided by sampling the `action_mask` returned in `info`.

    ## Episode End
    The episode ends if the following happens:

    - Termination:
            1. The taxi drops off the passenger.

    - Truncation (when using the time_limit wrapper):
            1. The length of the episode is 200.

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - p - transition probability for the state.
    - action_mask - if actions will cause a transition to a new state.

    For some cases, taking an action will have no effect on the state of the episode.
    In v0.25.0, ``info["action_mask"]`` contains a np.ndarray for each of the actions specifying
    if the action will change the state.

    To sample a modifying action, use ``action = env.action_space.sample(info["action_mask"])``
    Or with a Q-value based algorithm ``action = np.argmax(q_values[obs, np.where(info["action_mask"] == 1)[0]])``.

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('Taxi-v3')
    ```

    <a id="is_raining"></a>`is_raining=False`: If True the cab will move in intended direction with
    probability of 80% else will move in either left or right of target direction with
    equal probability of 10% in both directions.

    <a id="fickle_passenger"></a>`fickle_passenger=False`: If true the passenger has a 30% chance of changing
    destinations when the cab has moved one square away from the passenger's source location.  Passenger fickleness
    only happens on the first pickup and successful movement.  If the passenger is dropped off at the source location
    and picked up again, it is not triggered again.

    ## References
    <a id="taxi_ref"></a>[1] T. G. Dietterich, “Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition,”
    Journal of Artificial Intelligence Research, vol. 13, pp. 227–303, Nov. 2000, doi: 10.1613/jair.639.

    ## Version History
    * v3: Map Correction + Cleaner Domain Description, v0.25.0 action masking added to the reset and step information
        - In Gymnasium `1.2.0` the `is_rainy` and `fickle_passenger` arguments were added to align with Dietterich, 2000
    * v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
    * v1: Remove (3,2) from locs, add passidx<4 check
    * v0: Initial version release
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    initial_state_distrib: NDArray[np.float64]

    REWARD_COMPLETE = 20.0
    PENALTY_STEP = -1.0
    PENALTY_ILLEGAL_PICKUP_DROPOFF = -10.0

    max_row = 4
    max_col = 4

    desc = np.asarray(MAP, dtype="c")
    locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
    locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]

    # state is a combination of taxi row, col, passenger location, and destination
    num_states = (max_row + 1) * (max_col + 1) * len(Locations) * (len(Locations) - 1)

    action_space: spaces.Space[int] = spaces.Discrete(len(Actions))
    observation_space: spaces.Space[int] = spaces.Discrete(num_states)

    def _pickup(self, taxi_loc, pass_idx: Locations) -> tuple[Locations, float]:
        """Computes the new location and reward for pickup action."""
        if pass_idx != Locations.TAXI and taxi_loc == self.locs[pass_idx]:
            new_pass_idx = Locations.TAXI
            new_reward = self.PENALTY_STEP
        else:  # passenger not at location
            new_pass_idx = pass_idx
            new_reward = self.PENALTY_ILLEGAL_PICKUP_DROPOFF

        return new_pass_idx, new_reward

    def _dropoff(
        self, taxi_loc, pass_idx: Locations, dest_idx: Locations
    ) -> tuple[Locations, float, bool]:
        """Computes the new location and reward for return dropoff action."""
        if (taxi_loc == self.locs[dest_idx]) and pass_idx == Locations.TAXI:
            new_pass_idx = dest_idx
            new_terminated = True
            new_reward = self.REWARD_COMPLETE
        elif (taxi_loc in self.locs) and pass_idx == Locations.TAXI:
            new_pass_idx = Locations(self.locs.index(taxi_loc))
            new_terminated = False
            new_reward = self.PENALTY_STEP
        else:  # dropoff at wrong location
            new_pass_idx = pass_idx
            new_terminated = False
            new_reward = self.PENALTY_ILLEGAL_PICKUP_DROPOFF

        return new_pass_idx, new_reward, new_terminated

    def _build_movements(self) -> None:
        """Computes movements used in transitions and masks."""
        max_row = 4
        max_col = 4
        desc = self.desc
        # row, col, move direction, new position (row, col)
        self._movements = np.zeros((max_row + 1, max_col + 1, 4, 2), dtype=int)

        # Create coordinate grids for vectorized operations
        rows = np.arange(max_row + 1)[:, np.newaxis]  # shape (5, 1)
        cols = np.arange(max_col + 1)[np.newaxis, :]  # shape (1, 5)

        # All movements start at the current row or col
        self._movements[:, :, :, 0] = rows[:, :, np.newaxis]
        self._movements[:, :, :, 1] = cols[:, :, np.newaxis]

        # Check wall openings for all positions at once
        open_north = (desc[rows, 2 * cols + 1] != WALL_HORIZONTAL).astype(int)
        open_south = (desc[rows + 2, 2 * cols + 1] != WALL_HORIZONTAL).astype(int)
        open_east = (desc[rows + 1, 2 * cols + 2] != WALL_VERTICAL).astype(int)
        open_west = (desc[rows + 1, 2 * cols] != WALL_VERTICAL).astype(int)

        # If a space is open, move there
        self._movements[:, :, Actions.MOVE_EAST, 1] += open_east
        self._movements[:, :, Actions.MOVE_WEST, 1] -= open_west
        self._movements[:, :, Actions.MOVE_NORTH, 0] -= open_north
        self._movements[:, :, Actions.MOVE_SOUTH, 0] += open_south

    def _build_transitions(self, is_rainy: bool) -> None:
        """Computes the next action for a state (row, col, pass_idx, dest_idx) and action for `is_rainy`."""
        # mapping of left and right actions by action
        left = [
            Actions.MOVE_EAST,
            Actions.MOVE_WEST,
            Actions.MOVE_NORTH,
            Actions.MOVE_SOUTH,
        ]
        right = [
            Actions.MOVE_WEST,
            Actions.MOVE_EAST,
            Actions.MOVE_SOUTH,
            Actions.MOVE_NORTH,
        ]
        for row, col, pass_idx, dest_idx in itertools.product(
            range(self.max_row + 1),
            range(self.max_col + 1),
            Locations,
            Locations,
        ):
            if dest_idx == Locations.TAXI:
                continue

            state = self.encode(row, col, pass_idx, dest_idx)
            for action in Actions:
                new_row, new_col, new_pass_idx = row, col, pass_idx
                reward = (
                    self.PENALTY_STEP
                )  # default reward when there is no pickup/dropoff
                term = False

                if action in {
                    Actions.MOVE_SOUTH,
                    Actions.MOVE_NORTH,
                    Actions.MOVE_EAST,
                    Actions.MOVE_WEST,
                }:
                    new_row, new_col = self._movements[row, col, action]
                elif action == Actions.PICKUP:
                    new_pass_idx, reward = self._pickup((row, col), new_pass_idx)
                elif action == Actions.DROPOFF:
                    new_pass_idx, reward, term = self._dropoff(
                        (row, col), new_pass_idx, dest_idx
                    )
                new_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)

                # If it is rainy, veering left and right is possible only if the straight
                # ahead action causes a move.
                if is_rainy and (row != new_row or col != new_col):
                    l_row, l_col = self._movements[row, col, left[action]]
                    r_row, r_col = self._movements[row, col, right[action]]
                    left_state = self.encode(l_row, l_col, new_pass_idx, dest_idx)
                    right_state = self.encode(r_row, r_col, new_pass_idx, dest_idx)

                    prob_succeed = 0.8
                    prob_fail = 0.1
                    transitions = [
                        (prob_succeed, new_state, self.PENALTY_STEP, term),
                        (prob_fail, left_state, self.PENALTY_STEP, term),
                        (prob_fail, right_state, self.PENALTY_STEP, term),
                    ]
                else:
                    probability = 1.0
                    transitions = [(probability, new_state, reward, term)]
                self.P[state][action].extend(transitions)

    def _build_masks(self) -> None:
        """Computes movement part of actions masks."""
        n_rows, n_cols, _, _ = self._movements.shape

        # create coordinate grids for broadcasting
        self._masks = np.zeros((n_rows, n_cols, 4), dtype=np.int8)
        row_coords = np.arange(n_rows)[
            :, np.newaxis, np.newaxis
        ]  # shape (n_rows, 1, 1)
        col_coords = np.arange(n_cols)[
            np.newaxis, :, np.newaxis
        ]  # shape (1, n_cols, 1)

        # if a movement gives the same location, the move is not allowed
        self._masks = (
            (self._movements[:, :, :, 0] != row_coords)
            | (self._movements[:, :, :, 1] != col_coords)
        ).astype(np.int8)

    def __init__(
        self,
        render_mode: str | None = None,
        is_rainy: bool = False,
        fickle_passenger: bool = False,
    ):
        self.desc = np.asarray(MAP, dtype="c")

        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]

        num_states = 500
        num_rows = 5
        num_columns = 5
        self.max_row = num_rows - 1
        self.max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(num_states)
        num_actions = len(Actions)
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }

        self._build_movements()
        self._build_masks()
        self._build_transitions(is_rainy)

        self.render_mode = render_mode
        self.fickle_passenger = fickle_passenger
        self.fickle_step = self.fickle_passenger and self.np_random.random() < 0.3

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.imgs: dict[str, pygame.Surface] = {}
        self.taxi_orientation = "taxi_0"

    def encode(self, taxi_row, taxi_col, pass_loc: Locations, dest_idx: Locations):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i) -> tuple[int, int, Locations, Locations]:
        out = []
        out.append(Locations(i % 4))
        i = i // 4
        out.append(Locations(i % 5))
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def action_mask(self, state: int):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(len(Actions), dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        # Copy the movement part from the pre-computed masks
        mask[:4] = self._masks[taxi_row, taxi_col, :].copy()

        # Add the pickup, dropoff part
        if pass_loc != Locations.TAXI and (taxi_row, taxi_col) == self.locs[pass_loc]:
            mask[Actions.PICKUP] = 1
        if pass_loc == Locations.TAXI and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ):
            mask[Actions.DROPOFF] = 1
        return mask

    def step(self, a):
        transitions = self.P[self.s][a]
        if len(transitions) > 1:
            probabilities = [t[0] for t in transitions]
            i = categorical_sample(probabilities, self.np_random)
        else:
            i = 0
        p, s, r, t = transitions[i]
        self.lastaction = a

        # If we are in the fickle step, the passenger has been in the vehicle for at
        # least a step and this step the position changed
        if self.fickle_step:
            prev_row, prev_col, prev_pass_loc, prev_dest_idx = self.decode(self.s)
            taxi_row, taxi_col, pass_loc, _ = self.decode(s)
            if prev_pass_loc == Locations.TAXI and (
                taxi_row != prev_row or taxi_col != prev_col
            ):
                self.fickle_step = False
                possible_destinations = [
                    i for i in range(len(Locations) - 1) if i != prev_dest_idx
                ]
                dest_idx = self.np_random.choice(possible_destinations)
                s = self.encode(taxi_row, taxi_col, pass_loc, dest_idx)

        self.s = s

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return int(s), r, t, False, {"prob": p, "action_mask": self.action_mask(s)}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.fickle_step = self.fickle_passenger and self.np_random.random() < 0.3
        self.taxi_orientation = "taxi_0"

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def _load_imgs(self) -> None:
        """Load the images needed for rendering."""
        import pygame

        def load_image(filename: str) -> pygame.Surface:
            """Load and scale an image from the img directory."""
            file_path = path.join(path.dirname(__file__), "img", filename)
            return pygame.transform.scale(pygame.image.load(file_path), self.cell_size)

        self.imgs = {
            "taxi_0": load_image("cab_front.png"),
            "taxi_1": load_image("cab_rear.png"),
            "taxi_2": load_image("cab_right.png"),
            "taxi_3": load_image("cab_left.png"),
            "passenger": load_image("passenger.png"),
            "destination": load_image("hotel.png"),
            "median_left": load_image("gridworld_median_left.png"),
            "median_horiz": load_image("gridworld_median_horiz.png"),
            "median_right": load_image("gridworld_median_right.png"),
            "median_top": load_image("gridworld_median_top.png"),
            "median_vert": load_image("gridworld_median_vert.png"),
            "median_bottom": load_image("gridworld_median_bottom.png"),
            "background": load_image("taxi_background.png"),
        }

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        elif self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            elif mode == "rgb_array":
                self.window = pygame.Surface(WINDOW_SIZE)

        assert self.window is not None, (
            "Something went wrong with pygame. This should never happen."
        )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if not self.imgs:
            self._load_imgs()

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.imgs["background"], cell)
                wall_img = None
                if desc[y][x] == WALL_VERTICAL:
                    if y == 0 or desc[y - 1][x] != WALL_VERTICAL:
                        wall_img = "median_top"
                    elif y == desc.shape[0] - 1 or desc[y + 1][x] != WALL_VERTICAL:
                        wall_img = "median_bottom"
                    else:
                        wall_img = "median_vert"
                elif desc[y][x] == WALL_HORIZONTAL:
                    if x == 0 or desc[y][x - 1] != WALL_HORIZONTAL:
                        wall_img = "median_left"
                    elif x == desc.shape[1] - 1 or desc[y][x + 1] != WALL_HORIZONTAL:
                        wall_img = "median_right"
                    else:
                        wall_img = "median_horiz"

                if wall_img is not None:
                    self.window.blit(self.imgs[wall_img], cell)

        for cell, color in zip(self.locs, self.locs_colors, strict=True):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        if pass_idx != Locations.TAXI:
            self.window.blit(
                self.imgs["passenger"], self.get_surf_loc(self.locs[pass_idx])
            )

        if self.lastaction in [
            Actions.MOVE_SOUTH,
            Actions.MOVE_NORTH,
            Actions.MOVE_WEST,
            Actions.MOVE_EAST,
        ]:
            self.taxi_orientation = f"taxi_{self.lastaction}"
        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        dest_coords = (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2)
        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(self.imgs["destination"], dest_coords)
            self.window.blit(self.imgs[self.taxi_orientation], taxi_location)
        else:  # change blit order for overlapping appearance
            self.window.blit(self.imgs[self.taxi_orientation], taxi_location)
            self.window.blit(self.imgs["destination"], dest_coords)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def _render_text(self):
        desc = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx != Locations.TAXI:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def clone_state(self) -> TaxiState:
        """Returns a copy of the current episode state."""
        return TaxiState(
            s=self.s,
            lastaction=self.lastaction,
            fickle_step=self.fickle_step,
            taxi_orientation=self.taxi_orientation,
            np_random_state=self.np_random.bit_generator.state,
        )

    def restore_state(self, state: TaxiState) -> None:
        """Restores the environment to a previously saved state."""
        self.s = state.s
        self.lastaction = state.lastaction
        self.fickle_step = state.fickle_step
        self.taxi_orientation = state.taxi_orientation
        self.np_random.bit_generator.state = state.np_random_state

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


def _compute_initial_state_distrib() -> NDArray[np.float64]:
    """Compute the initial state distribution for the Taxi environment."""
    num_rows = TaxiEnv.max_row + 1
    num_columns = TaxiEnv.max_col + 1
    distrib = np.zeros(TaxiEnv.num_states)

    for row, col, pass_idx, dest_idx in itertools.product(
        range(num_rows),
        range(num_columns),
        Locations,
        Locations,
    ):
        if pass_idx != dest_idx:
            state = TaxiEnv.encode(row, col, pass_idx, dest_idx)
            distrib[state] += 1

    distrib /= distrib.sum()
    return distrib


# Initialize initial_state_distrib at module load time
TaxiEnv.initial_state_distrib = _compute_initial_state_distrib()


# Taxi rider from https://franuka.itch.io/rpg-asset-pack
# All other assets by Mel Tillery http://www.cyaneus.com/
