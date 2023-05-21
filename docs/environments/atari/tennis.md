---
title: Tennis
---

# Tennis

```{figure} ../../_static/videos/atari/tennis.gif
:width: 120px
:name: Tennis
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Tennis-v5")` |

For more Tennis variants with different observation and action spaces, see the variants section.

## Description

You control the orange player playing against a computer-controlled blue player. The game follows the rules of tennis.The first player to win at least 6 games with a margin of at least two games wins the match. If the score is tied at 6-6, the first player to go 2 games up wins the match.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=555)

## Actions

Tennis has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As Tennis uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

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

The scoring is as per the sport of tennis, played till one set.
For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=555).

## Variants

Tennis has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                     | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------------------|-------------|--------------|------------------------------|
| Tennis-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Tennis-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Tennis-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Tennis-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| TennisDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| TennisNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Tennis-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Tennis-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Tennis-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Tennis-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| TennisDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| TennisNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Tennis-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Tennis-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, 2]`          | `0`            | `[0, 1, 2, 3]`           | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
