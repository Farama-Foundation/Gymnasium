---
title: Berzerk
---

# Berzerk

```{figure} ../../_static/videos/atari/berzerk.gif
:width: 120px
:name: Berzerk
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Berzerk-v5")` |

For more Berzerk variants with different observation and action spaces, see the variants section.

## Description

You are stuck in a maze with evil robots. You must destroy them and avoid touching the walls of the maze, as this will kill you. You may be awarded extra lives after scoring a sufficient number of points, depending on the game mode.You may also be chased by an undefeatable enemy, Evil Otto, that you must avoid. Evil Otto does not appear in the default mode.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=866)

## Actions

Berzerk has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As Berzerk uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

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

You score points for destroying robots.
For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SystemID=2600&SoftwareID=866&itemTypeID=HTMLMANUAL).

## Variants

Berzerk has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                      | obs_type=   | frameskip=   | repeat_action_probability=   |
|-----------------------------|-------------|--------------|------------------------------|
| Berzerk-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Berzerk-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Berzerk-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Berzerk-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| BerzerkDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| BerzerkNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Berzerk-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Berzerk-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Berzerk-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Berzerk-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| BerzerkDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| BerzerkNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Berzerk-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Berzerk-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes           | Default Mode   | Available Difficulties   | Default Difficulty   |
|---------------------------|----------------|--------------------------|----------------------|
| `[1, ..., 9, 16, 17, 18]` | `1`            | `[0]`                    | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
