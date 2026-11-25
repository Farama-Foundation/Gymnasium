# Frozen Lake

![image](_static/videos/toy_text/frozen_lake.gif)

This environment is part of the [Toy Text environments](../toy_text.md) which contains general information about the environment.

|                   |                                   |
|-------------------|-----------------------------------|
| Action Space      | `Discrete(4)`                     |
| Observation Space | `Discrete(16)`                    |
| Max Episode Steps | `100`                             |
| Reward Threshold  | `0.7`                             |
| import            | `gymnasium.make("FrozenLake-v1")` |

Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
by walking over the frozen lake.
The player may not always move in the intended direction due to the slippery nature of the frozen lake.

## Description

The game starts with the player at location `[0,0]` of the frozen lake grid world with the
goal located at far extent of the world e.g. `[3,3]` for the 4x4 environment.

Holes in the ice are distributed in set locations when using a pre-determined map
or in random locations when a random map is generated.
Randomly generated worlds will always have a path to the goal.

The player makes moves until they reach the goal or fall in a hole.

The lake is slippery (unless disabled) so the player may move perpendicular
to the intended direction sometimes (see `is_slippery` in Argument section).

Pixel art assets are part of Farama Foundation’s shared environment art set.

## Action Space

The action space is `Discrete(4)`, an integer in `{0, 1, 2, 3}` indicating
which direction to move the player.

- 0: Move left
- 1: Move down
- 2: Move right
- 3: Move up

## Observation Space

The observation is a value representing the player’s current position as
`current_row * ncols + current_col` (where both the row and col start at 0).
Therefore, the observation is returned as an integer.

For example, the goal position in the 4x4 map can be calculated as follows: 3 \* 4 + 3 = 15.
The observation space is `Discrete(nrow * ncol)`, so the number of possible observations depends on the size of the map.

## Starting State

The episode starts with the player in state `[0]` (location [0, 0]).

## Rewards

Default reward schedule:

- Reach goal: +1
- Reach hole: 0
- Reach frozen: 0

See `reward_schedule` for reward customization in the Argument section.

## Episode End

The episode ends if the following happens:

- Termination:
  1. The player moves into a hole.
  2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).
- Truncation (using the time_limit wrapper):
  1. The length of the episode is 100 for `FrozenLake-v1` (4x4 map), 200 for `FrozenLake8x8-v1`.

## Information

`step()` and `reset()` return a dict with the following keys:

- `prob`: transition probability for the state, which depends on the `is_slippery` and `success_rate` parameters
  (always 1.0 on `reset()` or if `is_slippery=False`).

## Arguments

FrozenLake has five parameters:

```python
import gymnasium as gym
gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="4x4",
    is_slippery=True,
    success_rate=1.0/3.0,
    reward_schedule=(1, 0, 0)
)
```

* `desc=None`: Used to specify non-preloaded maps.
  If `desc=None` then `map_name` will be used. If both `desc` and `map_name` are
  `None` a random 8x8 map with 80% of locations frozen will be generated.

  To Specify a custom map - `desc=["SFFF", "FHFH", "FFFH", "HFFG"]`
  The tile letters denote:
  - “S” for Start tile
  - “G” for Goal tile
  - “F” for frozen tile
  - “H” for a tile with a hole

  A random generated map can be specified by calling the function `generate_random_map`.
  ```default
  from gymnasium.envs.toy_text.frozen_lake import generate_random_map

  gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
  ```
* `map_name="4x4"` - Helps load two predefined map names (`4x4` and `8x8`)
  ```default
  "4x4":[
      "SFFF",
      "FHFH",
      "FFFH",
      "HFFG"
  ]

  "8x8": [
      "SFFFFFFF",
      "FFFFFFFF",
      "FFFHFFFF",
      "FFFFFHFF",
      "FFFHFFFF",
      "FHHFFFHF",
      "FHFFHFHF",
      "FFFHFFFG",
  ]
  ```
* `is_slippery=True`: If true the player will move in intended direction with probability specified by the
  `success_rate` else will move in either perpendicular direction with equal probability in both directions.

  For example, if action is left, `is_slippery` is True, and `success_rate` is 1/3, then:
  - P(move left)=1/3
  - P(move up)=1/3
  - P(move down)=1/3

  If action is up, `is_slippery` is True, and `success_rate` is 3/4, then:
  - P(move up)=3/4
  - P(move left)=1/8
  - P(move right)=1/8
* `success_rate=1.0/3.0`: Used to specify the probability of moving in the intended direction when is_slippery=True
* `reward_schedule=(1, 0, 0)`: Used to specify reward amounts for reaching certain tiles.
  The indices correspond to: Reach Goal, Reach Hole, Reach Frozen (includes Start), Respectively

## Version History

* v1: Bug fixes to rewards (v1.3, added reward customization)
* v0: Initial version release
