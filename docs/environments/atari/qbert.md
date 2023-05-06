---
title: Qbert
---

# Qbert

```{figure} ../../_static/videos/atari/qbert.gif
:width: 120px
:name: Qbert
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(6) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Qbert-v5")` |

For more Qbert variants with different observation and action spaces, see the variants section.

## Description

You are Q*bert. Your goal is to change the color of all the cubes on the pyramid to the pyramid's 'destination' color. To do this, you must hop on each cube on the pyramid one at a time while avoiding nasty creatures that lurk there.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=1224)

## Actions

Qbert has the action space of `Discrete(6)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning   | Value   | Meaning   | Value   | Meaning   |
|---------|-----------|---------|-----------|---------|-----------|
| `0`     | `NOOP`    | `1`     | `FIRE`    | `2`     | `UP`      |
| `3`     | `RIGHT`   | `4`     | `LEFT`    | `5`     | `DOWN`    |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.

### Rewards

You score points for changing color of the cubes to their destination colors or by defeating enemies. You also gain points for completing a level.
For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SystemID=2600&SoftwareID=1224&itemTypeID=HTMLMANUAL).

## Variants

Qbert has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                    | obs_type=   | frameskip=   | repeat_action_probability=   |
|---------------------------|-------------|--------------|------------------------------|
| Qbert-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Qbert-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Qbert-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Qbert-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| QbertDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| QbertNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Qbert-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Qbert-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Qbert-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Qbert-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| QbertDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| QbertNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Qbert-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Qbert-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

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
