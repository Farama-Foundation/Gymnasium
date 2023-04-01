---
title: NameThisGame
---

# NameThisGame

```{figure} ../../_static/videos/atari/name_this_game.gif
:width: 120px
:name: NameThisGame
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(6) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/NameThisGame-v5")` |

For more NameThisGame variants with different observation and action spaces, see the variants section.

## Description

Your goal is to defend the treasure that you have discovered. You must fight off a shark and an octopus while keeping an eye on your oxygen supply.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=323)

## Actions

NameThisGame has the action space of `Discrete(6)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning   | Value   | Meaning     | Value   | Meaning    |
|---------|-----------|---------|-------------|---------|------------|
| `0`     | `NOOP`    | `1`     | `FIRE`      | `2`     | `RIGHT`    |
| `3`     | `LEFT`    | `4`     | `RIGHTFIRE` | `5`     | `LEFTFIRE` |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.


## Variants

NameThisGame has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                           | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------------------------|-------------|--------------|------------------------------|
| NameThisGame-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| NameThisGame-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| NameThisGame-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| NameThisGame-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| NameThisGameDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| NameThisGameNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| NameThisGame-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| NameThisGame-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| NameThisGame-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| NameThisGame-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| NameThisGameDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| NameThisGameNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/NameThisGame-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/NameThisGame-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[8, 24, 40]`     | `8`            | `[0, 1]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
