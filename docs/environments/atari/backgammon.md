---
title: Backgammon
---

# Backgammon

```{figure} ../../_static/videos/atari/backgammon.gif
:width: 120px
:name: Backgammon
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(3) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Backgammon-v5")` |

For more Backgammon variants with different observation and action spaces, see the variants section.

## Description

Your goal is to move all your pieces off the board (called 'bearing off'). Players take turns rolling dice and moving their pieces.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=12)

## Actions

Backgammon has the action space of `Discrete(3)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning   | Value   | Meaning   | Value   | Meaning   |
|---------|-----------|---------|-----------|---------|-----------|
| `0`     | `FIRE`    | `1`     | `RIGHT`   | `2`     | `LEFT`    |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.


## Variants

Backgammon has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                | obs_type=   | frameskip=   | repeat_action_probability=   |
|-----------------------|-------------|--------------|------------------------------|
| ALE/Backgammon-v5     | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Backgammon-ram-v5 | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0]`             | `0`            | `[3]`                    | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
