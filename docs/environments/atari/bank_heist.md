---
title: BankHeist
---

# BankHeist

```{figure} ../../_static/videos/atari/bank_heist.gif
:width: 120px
:name: BankHeist
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | Discrete(18) |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import | `gymnasium.make("ALE/BankHeist-v5")` |

For more BankHeist variants with different observation and action spaces, see the variants section.

## Description

You are a bank robber and (naturally) want to rob as many banks as possible. You control your getaway car and must navigate maze-like cities. The police chases you and will appear whenever you rob a bank. You may destroy police cars by dropping sticks of dynamite. You can fill up your gas tank by entering a new city.At the beginning of the game you have four lives. Lives are lost if you run out of gas, are caught by the police,or run over the dynamite you have previously dropped.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=1008)

## Actions

BankHeist has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As BankHeist uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

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

You score points for robbing banks and destroying police cars. If you rob nine or more banks, and then leave the city,
you will score extra points.
For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=1008).

## Variants

BankHeist has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                        | obs_type=   | frameskip=   | repeat_action_probability=   |
|-------------------------------|-------------|--------------|------------------------------|
| BankHeist-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| BankHeist-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| BankHeist-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| BankHeist-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| BankHeistDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| BankHeistNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| BankHeist-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| BankHeist-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| BankHeist-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| BankHeist-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| BankHeistDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| BankHeistNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/BankHeist-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/BankHeist-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes                 | Default Mode   | Available Difficulties   | Default Difficulty   |
|---------------------------------|----------------|--------------------------|----------------------|
| `[0, 4, 8, 12, 16, 20, 24, 28]` | `0`            | `[0, 1, 2, 3]`           | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
