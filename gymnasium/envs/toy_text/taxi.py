from contextlib import closing
from io import StringIO
from os import path

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled


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

    def _pickup(self, taxi_loc, pass_idx, reward):
        """Computes the new location and reward for pickup action."""
        if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
            new_pass_idx = 4
            new_reward = reward
        else:  # passenger not at location
            new_pass_idx = pass_idx
            new_reward = -10

        return new_pass_idx, new_reward

    def _dropoff(self, taxi_loc, pass_idx, dest_idx, default_reward):
        """Computes the new location and reward for return dropoff action."""
        if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
            new_pass_idx = dest_idx
            new_terminated = True
            new_reward = 20
        elif (taxi_loc in self.locs) and pass_idx == 4:
            new_pass_idx = self.locs.index(taxi_loc)
            new_terminated = False
            new_reward = default_reward
        else:  # dropoff at wrong location
            new_pass_idx = pass_idx
            new_terminated = False
            new_reward = -10

        return new_pass_idx, new_reward, new_terminated

    def _build_dry_transitions(self, row, col, pass_idx, dest_idx, action):
        """Computes the next action for a state (row, col, pass_idx, dest_idx) and action."""
        state = self.encode(row, col, pass_idx, dest_idx)

        taxi_loc = (row, col)
        new_row, new_col, new_pass_idx = row, col, pass_idx
        reward = -1  # default reward when there is no pickup/dropoff
        terminated = False

        if action == 0:
            new_row = min(row + 1, self.max_row)
        elif action == 1:
            new_row = max(row - 1, 0)
        if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
            new_col = min(col + 1, self.max_col)
        elif action == 3 and self.desc[1 + row, 2 * col] == b":":
            new_col = max(col - 1, 0)
        elif action == 4:  # pickup
            new_pass_idx, reward = self._pickup(taxi_loc, new_pass_idx, reward)
        elif action == 5:  # dropoff
            new_pass_idx, reward, terminated = self._dropoff(
                taxi_loc, new_pass_idx, dest_idx, reward
            )

        new_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)
        self.P[state][action].append((1.0, new_state, reward, terminated))

    def _calc_new_position(self, row, col, movement, offset=0):
        """Calculates the new position for a row and col to the movement."""
        dr, dc = movement
        new_row = max(0, min(row + dr, self.max_row))
        new_col = max(0, min(col + dc, self.max_col))
        if self.desc[1 + new_row, 2 * new_col + offset] == b":":
            return new_row, new_col
        else:  # Default to current position if not traversable
            return row, col

    def _build_rainy_transitions(self, row, col, pass_idx, dest_idx, action):
        """Computes the next action for a state (row, col, pass_idx, dest_idx) and action for `is_rainy`."""
        state = self.encode(row, col, pass_idx, dest_idx)

        taxi_loc = left_pos = right_pos = (row, col)
        new_row, new_col, new_pass_idx = row, col, pass_idx
        reward = -1  # default reward when there is no pickup/dropoff
        terminated = False

        moves = {
            0: ((1, 0), (0, -1), (0, 1)),  # Down
            1: ((-1, 0), (0, -1), (0, 1)),  # Up
            2: ((0, 1), (1, 0), (-1, 0)),  # Right
            3: ((0, -1), (1, 0), (-1, 0)),  # Left
        }

        # Check if movement is allowed
        if (
            action in {0, 1}
            or (action == 2 and self.desc[1 + row, 2 * col + 2] == b":")
            or (action == 3 and self.desc[1 + row, 2 * col] == b":")
        ):
            dr, dc = moves[action][0]
            new_row = max(0, min(row + dr, self.max_row))
            new_col = max(0, min(col + dc, self.max_col))

            left_pos = self._calc_new_position(row, col, moves[action][1], offset=2)
            right_pos = self._calc_new_position(row, col, moves[action][2])
        elif action == 4:  # pickup
            new_pass_idx, reward = self._pickup(taxi_loc, new_pass_idx, reward)
        elif action == 5:  # dropoff
            new_pass_idx, reward, terminated = self._dropoff(
                taxi_loc, new_pass_idx, dest_idx, reward
            )
        intended_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)

        if action <= 3:
            left_state = self.encode(left_pos[0], left_pos[1], new_pass_idx, dest_idx)
            right_state = self.encode(
                right_pos[0], right_pos[1], new_pass_idx, dest_idx
            )

            self.P[state][action].append((0.8, intended_state, -1, terminated))
            self.P[state][action].append((0.1, left_state, -1, terminated))
            self.P[state][action].append((0.1, right_state, -1, terminated))
        else:
            self.P[state][action].append((1.0, intended_state, reward, terminated))

    def __init__(
        self,
        render_mode: str | None = None,
        is_rainy: bool = False,
        fickle_passenger: bool = False,
    ):
        self.desc = np.asarray(MAP, dtype="c")

        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]

        num_states = 500
        num_rows = 5
        num_columns = 5
        self.max_row = num_rows - 1
        self.max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }

        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            if is_rainy:
                                self._build_rainy_transitions(
                                    row,
                                    col,
                                    pass_idx,
                                    dest_idx,
                                    action,
                                )
                            else:
                                self._build_dry_transitions(
                                    row,
                                    col,
                                    pass_idx,
                                    dest_idx,
                                    action,
                                )
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

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
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def action_mask(self, state: int):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(6, dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        if taxi_row < 4:
            mask[0] = 1
        if taxi_row > 0:
            mask[1] = 1
        if taxi_col < 4 and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b":":
            mask[2] = 1
        if taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b":":
            mask[3] = 1
        if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc == 4 and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ):
            mask[5] = 1
        return mask

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.lastaction = a

        shadow_row, shadow_col, shadow_pass_loc, shadow_dest_idx = self.decode(self.s)
        taxi_row, taxi_col, pass_loc, _ = self.decode(s)

        # If we are in the fickle step, the passenger has been in the vehicle for at least a step and this step the
        # position changed
        if (
            self.fickle_passenger
            and self.fickle_step
            and shadow_pass_loc == 4
            and (taxi_row != shadow_row or taxi_col != shadow_col)
        ):
            self.fickle_step = False
            possible_destinations = [
                i for i in range(len(self.locs)) if i != shadow_dest_idx
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
        self.taxi_orientation = 0

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

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

        assert (
            self.window is not None
        ), "Something went wrong with pygame. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        if pass_idx < 4:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[pass_idx]))

        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction
        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:  # change blit order for overlapping appearance
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )

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

        if pass_idx < 4:
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

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


# Taxi rider from https://franuka.itch.io/rpg-asset-pack
# All other assets by Mel Tillery http://www.cyaneus.com/
