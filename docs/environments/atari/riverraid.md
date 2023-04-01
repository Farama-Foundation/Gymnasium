---
title: Riverraid
---

# Riverraid

```{figure} ../../_static/videos/atari/riverraid.gif
:width: 120px
:name: Riverraid
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Riverraid-v5")` |

For more Riverraid variants with different observation and action spaces, see the variants section.

## Description

You control a jet that flies over a river: you can move it sideways and fire missiles to destroy enemy objects. Each time an enemy object is destroyed you score points (i.e. rewards).You lose a jet when you run out of fuel: fly over a fuel depot when you begin to run low.You lose a jet even when it collides with the river bank or one of the enemy objects (except fuel depots).The game begins with a squadron of three jets in reserve and you're given an additional jet (up to 9) for each 10,000 points you score.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=409)

## Actions

Riverraid has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As Riverraid uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

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

Score points are your only reward. You get score points each time you destroy an enemy object:

| Enemy Object | Score Points |
|--------------|--------------|
| Tanker       | 30           |
| Helicopter   | 60           |
| Fuel Depot   | 80           |
| Jet          | 100          |
| Bridge       | 500          |

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=409).

## Variants

Riverraid has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                        | obs_type=   | frameskip=   | repeat_action_probability=   |
|-------------------------------|-------------|--------------|------------------------------|
| Riverraid-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Riverraid-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Riverraid-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Riverraid-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| RiverraidDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| RiverraidNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Riverraid-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Riverraid-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Riverraid-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Riverraid-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| RiverraidDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| RiverraidNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Riverraid-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Riverraid-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

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
