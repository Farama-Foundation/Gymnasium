---
title: VideoCube
---

# VideoCube

```{figure} ../../_static/videos/atari/video_cube.gif
:width: 120px
:name: VideoCube
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/VideoCube-v5")` |

For more VideoCube variants with different observation and action spaces, see the variants section.

## Description

Solve a Rubik's cube in a nonstandard way: guide Hubie around the cube and swap tiles on the cubes face with one another until each face consists of only one color.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=974)

## Actions

VideoCube has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As VideoCube uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

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


## Variants

VideoCube has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id               | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------------|-------------|--------------|------------------------------|
| ALE/VideoCube-v5     | `"rgb"`     | `4`          | `0.25`                       |
| ALE/VideoCube-ram-v5 | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Default Mode   | Available Difficulties   | Default Difficulty   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|--------------------------|----------------------|
| `[0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302, 400, 401, 402, 500, 501, 502, 600, 601, 602, 700, 701, 702, 800, 801, 802, 900, 901, 902, 1000, 1001, 1002, 1100, 1101, 1102, 1200, 1201, 1202, 1300, 1301, 1302, 1400, 1401, 1402, 1500, 1501, 1502, 1600, 1601, 1602, 1700, 1701, 1702, 1800, 1801, 1802, 1900, 1901, 1902, 2000, 2001, 2002, 2100, 2101, 2102, 2200, 2201, 2202, 2300, 2301, 2302, 2400, 2401, 2402, 2500, 2501, 2502, 2600, 2601, 2602, 2700, 2701, 2702, 2800, 2801, 2802, 2900, 2901, 2902, 3000, 3001, 3002, 3100, 3101, 3102, 3200, 3201, 3202, 3300, 3301, 3302, 3400, 3401, 3402, 3500, 3501, 3502, 3600, 3601, 3602, 3700, 3701, 3702, 3800, 3801, 3802, 3900, 3901, 3902, 4000, 4001, 4002, 4100, 4101, 4102, 4200, 4201, 4202, 4300, 4301, 4302, 4400, 4401, 4402, 4500, 4501, 4502, 4600, 4601, 4602, 4700, 4701, 4702, 4800, 4801, 4802, 4900, 4901, 4902, 5000, 5001, 5002]` | `0`            | `[0, 1]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
