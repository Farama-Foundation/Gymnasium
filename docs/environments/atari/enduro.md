---
title: Enduro
---

# Enduro

```{figure} ../../_static/videos/atari/enduro.gif
:width: 120px
:name: Enduro
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(9) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Enduro-v5")` |

For more Enduro variants with different observation and action spaces, see the variants section.

## Description

You are a racer in the National Enduro, a long-distance endurance race. You must overtake a certain amount of cars each day to stay on the race. The first day you need to pass 200 cars, and 300 foreach following day. The game ends if you do not meet your overtake quota for the day.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=163)

## Actions

Enduro has the action space of `Discrete(9)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning    | Value   | Meaning     | Value   | Meaning     |
|---------|------------|---------|-------------|---------|-------------|
| `0`     | `NOOP`     | `1`     | `FIRE`      | `2`     | `RIGHT`     |
| `3`     | `LEFT`     | `4`     | `DOWN`      | `5`     | `DOWNRIGHT` |
| `6`     | `DOWNLEFT` | `7`     | `RIGHTFIRE` | `8`     | `LEFTFIRE`  |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.

### Rewards

You get 1 point for each vehicle you overtake.

## Variants

Enduro has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                     | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------------------|-------------|--------------|------------------------------|
| Enduro-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Enduro-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Enduro-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Enduro-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| EnduroDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| EnduroNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Enduro-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Enduro-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Enduro-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Enduro-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| EnduroDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| EnduroNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Enduro-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Enduro-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0]`             | `0`            | `[0]`                    | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
