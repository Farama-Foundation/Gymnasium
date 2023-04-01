---
title: DoubleDunk
---

# DoubleDunk

```{figure} ../../_static/videos/atari/double_dunk.gif
:width: 120px
:name: DoubleDunk
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/DoubleDunk-v5")` |

For more DoubleDunk variants with different observation and action spaces, see the variants section.

## Description

You are playing a 2v2 game of basketball. At the start of each possession, you select between a set of different plays and then execute them to either score or prevent your rivals from scoring.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=153)

## Actions

DoubleDunk has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As DoubleDunk uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

| Value   | Meaning      | Value   | Meaning         | Value   | Meaning        |
|---------|--------------|---------|-----------------|---------|----------------|
| `0`     | `NOOP`       | `1`     | `FIRE`          | `2`     | `UP`           |
| `3`     | `RIGHT`      | `4`     | `LEFT`          | `5`     | `DOWN`         |
| `6`     | `UPRIGHT`    | `7`     | `UPLEFT`        | `8`     | `DOWNRIGHT`    |
| `9`     | `DOWNLEFT`   | `10`    | `UPFIRE`        | `11`    | `RIGHTFIRE`    |
| `12`    | `LEFTFIRE`   | `13`    | `DOWNFIRE`      | `14`    | `UPRIGHTFIRE`  |
| `15`    | `UPLEFTFIRE` | `16`    | `DOWNRIGHTFIRE` | `17`    | `DOWNLEFTFIRE` |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.

### Rewards

Scores follow the rules of basketball. You can get either 3 points, 2 points foul line) depending
from where you shoot. After a defensive foul, a successful shot from the foul line gives you 1
point.

## Variants

DoubleDunk has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                         | obs_type=   | frameskip=   | repeat_action_probability=   |
|--------------------------------|-------------|--------------|------------------------------|
| DoubleDunk-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| DoubleDunk-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| DoubleDunk-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| DoubleDunk-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| DoubleDunkDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| DoubleDunkNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| DoubleDunk-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| DoubleDunk-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| DoubleDunk-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| DoubleDunk-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| DoubleDunkDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| DoubleDunkNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/DoubleDunk-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/DoubleDunk-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, ..., 15]`    | `0`            | `[0]`                    | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
