---
title: TicTacToe3D
---

# TicTacToe3D

```{figure} ../../_static/videos/atari/tic_tac_toe_3d.gif
:width: 120px
:name: TicTacToe3D
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(10) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/TicTacToe3D-v5")` |

For more TicTacToe3D variants with different observation and action spaces, see the variants section.

## Description

Players take turns placing their mark (an X or an O) on a 3-dimensional, 4 x 4 x 4 grid in an attempt to get 4 in a row before their opponent does.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=521)

## Actions

TicTacToe3D has the action space of `Discrete(10)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning    | Value   | Meaning   | Value   | Meaning     |
|---------|------------|---------|-----------|---------|-------------|
| `0`     | `NOOP`     | `1`     | `FIRE`    | `2`     | `UP`        |
| `3`     | `RIGHT`    | `4`     | `LEFT`    | `5`     | `DOWN`      |
| `6`     | `UPRIGHT`  | `7`     | `UPLEFT`  | `8`     | `DOWNRIGHT` |
| `9`     | `DOWNLEFT` |         |           |         |             |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.


## Variants

TicTacToe3D has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                 | obs_type=   | frameskip=   | repeat_action_probability=   |
|------------------------|-------------|--------------|------------------------------|
| ALE/TicTacToe3D-v5     | `"rgb"`     | `4`          | `0.25`                       |
| ALE/TicTacToe3D-ram-v5 | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, ..., 8]`     | `0`            | `[0, 2]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
