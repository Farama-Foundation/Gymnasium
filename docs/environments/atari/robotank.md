---
title: Robotank
---

# Robotank

```{figure} ../../_static/videos/atari/robotank.gif
:width: 120px
:name: Robotank
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Robotank-v5")` |

For more Robotank variants with different observation and action spaces, see the variants section.

## Description

You control your Robot Tanks to destroy enemies and avoid enemy fire.Game ends when all of your Robot Tanks are destroyed or all 12 enemy squadrons are destroyed.The game begins with one active Robot Tank and three reserves.Your Robot Tank may get lost when it is hit by enemy    rocket fire - your video scrambles with static interference when this    happens - or just become damaged - sensors report the damage by flashing on your control panel (look at V/C/R/T squares).You earn one bonus Robot Tank for every enemy squadron destroyed. The maximum number of bonus Robot Tanks allowed at any one time is 12.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=416)

## Actions

Robotank has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As Robotank uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

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

The number of enemies destroyed is the only reward.

A small tank appears at the top of your screen for each enemy
   you destroy.  A square with the number 12 appears each time a squadron of twelve enemies are
   destroyed.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=416).

## Variants

Robotank has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                       | obs_type=   | frameskip=   | repeat_action_probability=   |
|------------------------------|-------------|--------------|------------------------------|
| Robotank-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Robotank-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Robotank-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Robotank-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| RobotankDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| RobotankNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Robotank-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Robotank-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Robotank-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Robotank-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| RobotankDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| RobotankNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Robotank-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Robotank-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

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
