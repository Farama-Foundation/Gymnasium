---
title: TimePilot
---

# TimePilot

```{figure} ../../_static/videos/atari/time_pilot.gif
:width: 120px
:name: TimePilot
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(10) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/TimePilot-v5")` |

For more TimePilot variants with different observation and action spaces, see the variants section.

## Description

You control an aircraft. Use it to destroy your enemies. As you progress in the game, you encounter enemies with technology that is increasingly from the future.

For a more detailed documentation, see [the AtariAge page](http://www.atarimania.com/game-atari-2600-vcs-time-pilot_8038.html)

## Actions

TimePilot has the action space of `Discrete(10)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning    | Value   | Meaning     | Value   | Meaning    |
|---------|------------|---------|-------------|---------|------------|
| `0`     | `NOOP`     | `1`     | `FIRE`      | `2`     | `UP`       |
| `3`     | `RIGHT`    | `4`     | `LEFT`      | `5`     | `DOWN`     |
| `6`     | `UPFIRE`   | `7`     | `RIGHTFIRE` | `8`     | `LEFTFIRE` |
| `9`     | `DOWNFIRE` |         |             |         |            |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.

### Rewards

You score points for destroying enemies, gaining more points for difficult enemies.
For a more detailed documentation, see [the Atari Mania page](http://www.atarimania.com/game-atari-2600-vcs-time-pilot_8038.html).

## Variants

TimePilot has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                        | obs_type=   | frameskip=   | repeat_action_probability=   |
|-------------------------------|-------------|--------------|------------------------------|
| TimePilot-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| TimePilot-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| TimePilot-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| TimePilot-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| TimePilotDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| TimePilotNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| TimePilot-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| TimePilot-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| TimePilot-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| TimePilot-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| TimePilotDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| TimePilotNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/TimePilot-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/TimePilot-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0]`             | `0`            | `[0, 1, 2]`              | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
