---
title: AirRaid
---

# AirRaid

```{figure} ../../_static/videos/atari/air_raid.gif
:width: 120px
:name: AirRaid
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(6) |
| Observation Space | Box(0, 255, (250, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/AirRaid-v5")` |

For more AirRaid variants with different observation and action spaces, see the variants section.

## Description

You control a ship that can move sideways. You must protect two buildings (one on the right and one on the left side of the screen) from flying saucers that are trying to drop bombs on them.

## Actions

AirRaid has the action space of `Discrete(6)` with the table below listing the meaning of each action's meanings.
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

AirRaid has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                      | obs_type=   | frameskip=   | repeat_action_probability=   |
|-----------------------------|-------------|--------------|------------------------------|
| AirRaid-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| AirRaid-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| AirRaid-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| AirRaid-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| AirRaidDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| AirRaidNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| AirRaid-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| AirRaid-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| AirRaid-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| AirRaid-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| AirRaidDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| AirRaidNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/AirRaid-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/AirRaid-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[1, ..., 8]`     | `1`            | `[0]`                    | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
