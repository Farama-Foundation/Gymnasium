---
title: Jamesbond
---

# Jamesbond

```{figure} ../../_static/videos/atari/jamesbond.gif
:width: 120px
:name: Jamesbond
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Jamesbond-v5")` |

For more Jamesbond variants with different observation and action spaces, see the variants section.

## Description

Your mission is to control Mr. Bond's specially designed multipurpose craft to complete a variety of missions.The craft moves forward with a right motion and slightly back with a left motion.An up or down motion causes the craft to jump or dive.You can also fire by either lobbing a bomb to the bottom of the screen or firing a fixed angle shot to the top of the screen.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=250)

## Actions

Jamesbond has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As Jamesbond uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

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

The game ends when you complete the last mission or when you lose the last craft. In either case, you'll receive your final score.
There will be a rating based on your score. The highest rating in NOVICE is 006. The highest rating in AGENT is 007.
For a more detailed documentation, consult [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=250).

## Variants

Jamesbond has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                        | obs_type=   | frameskip=   | repeat_action_probability=   |
|-------------------------------|-------------|--------------|------------------------------|
| Jamesbond-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Jamesbond-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Jamesbond-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Jamesbond-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| JamesbondDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| JamesbondNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Jamesbond-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Jamesbond-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Jamesbond-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Jamesbond-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| JamesbondDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| JamesbondNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Jamesbond-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Jamesbond-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, 1]`          | `0`            | `[0]`                    | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
