---
title: JourneyEscape
---

# JourneyEscape

```{figure} ../../_static/videos/atari/journey_escape.gif
:width: 120px
:name: JourneyEscape
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(16) |
| Observation Space | Box(0, 255, (230, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/JourneyEscape-v5")` |

For more JourneyEscape variants with different observation and action spaces, see the variants section.

## Description

You must lead all 5 members of JOURNEY through waves of pesky characters and backstage obstacles to the Scarab Escape Vehicle before time runs out.You must also protect $50,000 in concert cash from grasping groupies, photographers, and promoters.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=252)

## Actions

JourneyEscape has the action space of `Discrete(16)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning        | Value   | Meaning      | Value   | Meaning         |
|---------|----------------|---------|--------------|---------|-----------------|
| `0`     | `NOOP`         | `1`     | `UP`         | `2`     | `RIGHT`         |
| `3`     | `LEFT`         | `4`     | `DOWN`       | `5`     | `UPRIGHT`       |
| `6`     | `UPLEFT`       | `7`     | `DOWNRIGHT`  | `8`     | `DOWNLEFT`      |
| `9`     | `RIGHTFIRE`    | `10`    | `LEFTFIRE`   | `11`    | `DOWNFIRE`      |
| `12`    | `UPRIGHTFIRE`  | `13`    | `UPLEFTFIRE` | `14`    | `DOWNRIGHTFIRE` |
| `15`    | `DOWNLEFTFIRE` |         |              |         |                 |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.

### Rewards

At the start of the game, you will have $50,000 and 60 units of time.
Your end game score with be dependent on how much time you have remaining and who you encounter along the way.
For a more detailed documentation, consult [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=252).

## Variants

JourneyEscape has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                            | obs_type=   | frameskip=   | repeat_action_probability=   |
|-----------------------------------|-------------|--------------|------------------------------|
| JourneyEscape-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| JourneyEscape-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| JourneyEscape-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| JourneyEscape-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| JourneyEscapeDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| JourneyEscapeNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| JourneyEscape-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| JourneyEscape-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| JourneyEscape-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| JourneyEscape-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| JourneyEscapeDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| JourneyEscapeNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/JourneyEscape-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/JourneyEscape-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0]`             | `0`            | `[0, 1]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
