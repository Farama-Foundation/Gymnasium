---
title: Pacman
---

# Pacman

```{figure} ../../_static/videos/atari/pacman.gif
:width: 120px
:name: Pacman
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(5) |
| Observation Space | Box(0, 255, (250, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Pacman-v5")` |

For more Pacman variants with different observation and action spaces, see the variants section.

## Description

A classic arcade game. Move Pac Man around a maze collecting food and avoiding ghosts- unless you eat a Power Pellet, then you can eat the ghosts too!

## Actions

Pacman has the action space of `Discrete(5)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning   | Value   | Meaning   | Value   | Meaning   |
|---------|-----------|---------|-----------|---------|-----------|
| `0`     | `NOOP`    | `1`     | `UP`      | `2`     | `RIGHT`   |
| `3`     | `LEFT`    | `4`     | `DOWN`    |         |           |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.


## Variants

Pacman has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                       | obs_type=   | frameskip=   | repeat_action_probability=   |
|------------------------------|-------------|--------------|------------------------------|
| MsPacman-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| MsPacman-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| MsPacman-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| MsPacman-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| MsPacmanDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| MsPacmanNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| MsPacman-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| MsPacman-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| MsPacman-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| MsPacman-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| MsPacmanDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| MsPacmanNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/MsPacman-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/MsPacman-ram-v5          | `"ram"`     | `4`          | `0.25`                       |
| ALE/Pacman-v5                | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Pacman-ram-v5            | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, ..., 7]`     | `0`            | `[0, 1]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
