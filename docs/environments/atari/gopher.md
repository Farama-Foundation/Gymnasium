---
title: Gopher
---

# Gopher

```{figure} ../../_static/videos/atari/gopher.gif
:width: 120px
:name: Gopher
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(8) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Gopher-v5")` |

For more Gopher variants with different observation and action spaces, see the variants section.

## Description

The player controls a shovel-wielding farmer who protects a crop of three carrots from a gopher.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=218)

## Actions

Gopher has the action space of `Discrete(8)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning     | Value   | Meaning    | Value   | Meaning   |
|---------|-------------|---------|------------|---------|-----------|
| `0`     | `NOOP`      | `1`     | `FIRE`     | `2`     | `UP`      |
| `3`     | `RIGHT`     | `4`     | `LEFT`     | `5`     | `UPFIRE`  |
| `6`     | `RIGHTFIRE` | `7`     | `LEFTFIRE` |         |           |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.

### Rewards

The exact reward dynamics depend on the environment and are usually documented in the game's manual. You can
find these manuals on [AtariAge](https://atariage.com/manual_html_page.php?SoftwareLabelID=218).

Atari environments are simulated via the Arcade Learning Environment (ALE) [[1]](#1).
## Variants

Gopher has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                     | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------------------|-------------|--------------|------------------------------|
| Gopher-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Gopher-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Gopher-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Gopher-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| GopherDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| GopherNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Gopher-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Gopher-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Gopher-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Gopher-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| GopherDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| GopherNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Gopher-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Gopher-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, 2]`          | `0`            | `[0, 1]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
