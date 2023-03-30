---
title: RoadRunner
---

# RoadRunner

```{figure} ../../_static/videos/atari/road_runner.gif
:width: 120px
:name: RoadRunner
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/RoadRunner-v5")` |

For more RoadRunner variants with different observation and action spaces, see the variants section.

## Description

You control the Road Runner(TM) in a race; you can control the direction to run in and times to jumps.The goal is to outrun Wile E. Coyote(TM) while avoiding the hazards of the desert.The game begins with three lives.  You lose a life when the coyote catches you, picks you up in a rocket, or shoots you with a cannon.  You also lose a life when a truck hits you, you hit a land mine, you fall off a cliff,or you get hit by a falling rock.You score points (i.e. rewards) by eating seeds along the road, eating steel shot, and destroying the coyote.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=412)

## Actions

RoadRunner has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As RoadRunner uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

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

Score points are your only reward. You get score points each time you:

| actions                                               | points |
|-------------------------------------------------------|--------|
| eat a pile of birdseed                                | 100    |
| eat steel shot                                        | 100    |
| get the coyote hit by a mine (cannonball, rock, etc.) | 200    |
| get the coyote hit by a truck                         | 1000   |

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=412).

## Variants

RoadRunner has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                         | obs_type=   | frameskip=   | repeat_action_probability=   |
|--------------------------------|-------------|--------------|------------------------------|
| RoadRunner-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| RoadRunner-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| RoadRunner-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| RoadRunner-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| RoadRunnerDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| RoadRunnerNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| RoadRunner-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| RoadRunner-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| RoadRunner-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| RoadRunner-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| RoadRunnerDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| RoadRunnerNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/RoadRunner-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/RoadRunner-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

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
