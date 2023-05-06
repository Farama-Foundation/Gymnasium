---
title: Seaquest
---

# Seaquest

```{figure} ../../_static/videos/atari/seaquest.gif
:width: 120px
:name: Seaquest
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/Seaquest-v5")` |

For more Seaquest variants with different observation and action spaces, see the variants section.

## Description

You control a sub able to move in all directions and fire torpedoes.The goal is to retrieve as many divers as you can, while dodging and blasting enemy subs and killer sharks; points will be awarded accordingly.The game begins with one sub and three waiting on the horizon. Each time you increase your score by 10,000 points, an extra sub will be delivered to yourbase.  You can only have six reserve subs on the screen at one time.Your sub will explode if it collides with anything except your own divers.The sub has a limited amount of oxygen that decreases at a constant rate during the game. When the oxygen tank is almost empty, you need to surface and if you don't do it in time, your sub will blow up and you'll lose one diver.  Each time you're forced to surface, with less than six divers, you lose one diver as well.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=424)

## Actions

Seaquest has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As Seaquest uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

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

Score points are your only reward.

Blasting enemy sub and killer shark is worth
20 points.  Every time you surface with six divers, the value of enemy subs
and killer sharks increases by 10, up to a maximum of 90 points each.

Rescued divers start at 50 points each.  Then, their point value increases by 50, every
time you surface, up to a maximum of 1000 points each.

You'll be further rewarded with bonus points for all the oxygen you have remaining the
moment you surface.  The more oxygen you have left, the more bonus points
you're given.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=424).

## Variants

Seaquest has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                       | obs_type=   | frameskip=   | repeat_action_probability=   |
|------------------------------|-------------|--------------|------------------------------|
| Seaquest-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Seaquest-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Seaquest-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Seaquest-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| SeaquestDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| SeaquestNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Seaquest-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Seaquest-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Seaquest-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Seaquest-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| SeaquestDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| SeaquestNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Seaquest-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Seaquest-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

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
