---
title: KungFuMaster
---

# KungFuMaster

```{figure} ../../_static/videos/atari/kung_fu_master.gif
:width: 120px
:name: KungFuMaster
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(14) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/KungFuMaster-v5")` |

For more KungFuMaster variants with different observation and action spaces, see the variants section.

## Description

You are a Kung-Fu Master fighting your way through the Evil Wizard's temple. Your goal is to rescue Princess Victoria, defeating various enemies along the way.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=268)

## Actions

KungFuMaster has the action space of `Discrete(14)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning         | Value   | Meaning        | Value   | Meaning      |
|---------|-----------------|---------|----------------|---------|--------------|
| `0`     | `NOOP`          | `1`     | `UP`           | `2`     | `RIGHT`      |
| `3`     | `LEFT`          | `4`     | `DOWN`         | `5`     | `DOWNRIGHT`  |
| `6`     | `DOWNLEFT`      | `7`     | `RIGHTFIRE`    | `8`     | `LEFTFIRE`   |
| `9`     | `DOWNFIRE`      | `10`    | `UPRIGHTFIRE`  | `11`    | `UPLEFTFIRE` |
| `12`    | `DOWNRIGHTFIRE` | `13`    | `DOWNLEFTFIRE` |         |              |

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.


## Variants

KungFuMaster has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                           | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------------------------|-------------|--------------|------------------------------|
| KungFuMaster-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| KungFuMaster-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| KungFuMaster-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| KungFuMaster-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| KungFuMasterDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| KungFuMasterNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| KungFuMaster-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| KungFuMaster-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| KungFuMaster-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| KungFuMaster-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| KungFuMasterDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| KungFuMasterNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/KungFuMaster-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/KungFuMaster-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

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
