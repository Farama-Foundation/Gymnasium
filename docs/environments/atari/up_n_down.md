---
title: UpNDown
---

# UpNDown

```{figure} ../../_static/videos/atari/up_n_down.gif
:width: 120px
:name: UpNDown
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(6) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/UpNDown-v5")` |

For more UpNDown variants with different observation and action spaces, see the variants section.

## Description

Your goal is to steer your baja bugger to collect prizes and eliminate opponents.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=574)

## Actions

UpNDown has the action space of `Discrete(6)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning   | Value   | Meaning   | Value   | Meaning    |
|---------|-----------|---------|-----------|---------|------------|
| `0`     | `NOOP`    | `1`     | `FIRE`    | `2`     | `UP`       |
| `3`     | `DOWN`    | `4`     | `UPFIRE`  | `5`     | `DOWNFIRE` |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.


## Variants

UpNDown has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                      | obs_type=   | frameskip=   | repeat_action_probability=   |
|-----------------------------|-------------|--------------|------------------------------|
| UpNDown-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| UpNDown-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| UpNDown-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| UpNDown-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| UpNDownDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| UpNDownNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| UpNDown-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| UpNDown-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| UpNDown-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| UpNDown-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| UpNDownDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| UpNDownNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/UpNDown-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/UpNDown-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0]`             | `0`            | `[0, 1, 2, 3]`           | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
