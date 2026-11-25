# Taxi

![image](_static/videos/toy_text/taxi.gif)

This environment is part of the [Toy Text environments](../toy_text.md) which contains general information about the environment.

|                   |                             |
|-------------------|-----------------------------|
| Action Space      | `Discrete(6)`               |
| Observation Space | `Discrete(500)`             |
| Max Episode Steps | `200`                       |
| Reward Threshold  | `8`                         |
| import            | `gymnasium.make("Taxi-v4")` |

The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them
off at one of four locations.

## Description

There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the
5x5 grid world. The taxi starts off at a random square and the passenger at one of the
designated locations.

The goal is to move the taxi to the passenger’s location, pick up the passenger,
move to the passenger’s desired destination, and
drop off the passenger. Once the passenger is dropped off, the episode ends.

The player receives positive rewards for successfully dropping-off the passenger at the correct
location. Negative rewards for incorrect attempts to pick-up/drop-off passenger and
for each step where another reward is not received.

Map:

```none
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

From “Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition”
by Tom Dietterich [<a href="#taxi_ref">1</a>].

## Action Space

The action space is `Discrete(6)`, an integer in `{0, 1, 2, 3, 4, 5}` indicating
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

Destination on the ansi rendered map are represented with the first letter of the color.

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
and 3 destinations (excluding the passenger’s current location).

## Rewards

- -1 per step unless another reward is triggered.
- +20 delivering passenger.
- -10  executing “pickup” and “drop-off” actions illegally.

Picking up is illegal unless the taxi is at the passenger’s location, and dropping off is illegal unless the
passenger is in the taxi and the taxi is at one of the four designated locations. Dropping the passenger off at
a designated location other than the destination is legal: it incurs the -1 step penalty and leaves the
passenger at that location.

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

- “prob”: transition probability for the state.
- “action_mask”: if actions will cause a transition to a new state. This was added in v0.25.0

For some cases, taking an action will have no effect on the state of the episode.
In v0.25.0, `info["action_mask"]` contains a np.ndarray for each of the actions specifying
if the action will change the state.

To sample a modifying action, use `action = env.action_space.sample(mask=info["action_mask"])`.
Or with a Q-value based algorithm, `valid = np.where(info["action_mask"] == 1)[0]` then
`action = valid[np.argmax(q_values[obs, valid])]`.

## Arguments

```python
import gymnasium as gym
gym.make('Taxi-v4', is_rainy=False, rainy_probability=0.8, fickle_passenger=False, fickle_probability=0.3)
```

<a id="is_rainy"></a>`is_rainy=False`: If True the cab will move in the intended direction with probability
80%, controlled by `rainy_probability`, else in a lateral direction with equal probability.
If the intended move is blocked by a wall or the grid boundary, the cab does not move at all.
Pickup and dropoff actions remain deterministic (probability 1.0).

<a id="rainy_probability"></a>`rainy_probability=0.8`: When `is_rainy=True`, the probability of
moving in the intended direction. Each lateral direction is given `(1 - rainy_probability) / 2`.

<a id="fickle_passenger"></a>`fickle_passenger=False`: If True, then on reset the passenger has a 30% chance,
controlled by `fickle_probability`, of being fickle. A fickle passenger changes destination (uniformly to one
of the other three locations) the first time the cab moves to a new square with the passenger on board.
This happens at most once per episode.

<a id="fickle_probability"></a>`fickle_probability=0.3`: When `fickle_passenger=True`, the probability
(sampled on reset) that the passenger changes destination on the first move after pickup.

## References

<a id="taxi_ref"></a>[1] T. G. Dietterich, “Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition,”
Journal of Artificial Intelligence Research, vol. 13, pp. 227–303, Nov. 2000, doi: 10.1613/jair.639.

## Version History

* v4: In v1.3.0, fix `is_rainy=True` and `fickle_passenger=True` implementations
  - Add `rainy_probability` and `fickle_probability` arguments to tune the stochastic behaviour
* v3: Map Correction + Cleaner Domain Description,
  - In v0.25.0 action masking added to the reset and step information
  - In v1.2.0 added `is_rainy` and `fickle_passenger` arguments to align with Dietterich, 2000 Section 7.1
* v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
* v1: Remove (3,2) from locs, add passidx<4 check
* v0: Initial version release
