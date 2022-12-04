---
title: Bank Heist
---

# Bank Heist

```{figure} ../../_static/videos/atari/bank_heist.gif
:width: 120px
:name: Bank Heist
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                      |
|-------------------|--------------------------------------|
| Action Space      | Discrete(18)                         |
| Observation Space | (210, 160, 3)                        |
| Observation High  | 255                                  |
| Observation Low   | 0                                    |
| Import            | `gymnasium.make("ALE/BankHeist-v5")` |

## Description

You are a bank robber and (naturally) want to rob as many banks as possible. You control your getaway car and must
navigate maze-like cities. The police chases you and will appear whenever you rob a bank. You may destroy police cars
by dropping sticks of dynamite. You can fill up your gas tank by entering a new city.
At the beginning of the game you have four lives. Lives are lost if you run out of gas, are caught by the police,
or run over the dynamite you have previously dropped.
Detailed documentation can be found on [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=1008).

## Actions

By default, all actions that can be performed on an Atari 2600 are available in this environment.
Even if you use v0 or v4 or specify `full_action_space=False` during initialization, all actions
will be available in the default flavor.

## Observations

By default, the environment returns the RGB image that is displayed to human players as an observation. However, it is
possible to observe

- The 128 Bytes of RAM of the console
- A grayscale image

instead. The respective observation spaces are

- `Box([0 ... 0], [255 ... 255], (128,), uint8)`
- `Box([[0 ... 0]
 ...
 [0  ... 0]], [[255 ... 255]
 ...
 [255  ... 255]], (250, 160), uint8)
`

respectively. The general article on Atari environments outlines different ways to instantiate corresponding environments
via `gymnasium.make`.

### Rewards

You score points for robbing banks and destroying police cars. If you rob nine or more banks, and then leave the city,
you will score extra points.
For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=1008).

## Arguments

```python
env = gymnasium.make("ALE/BankHeist-v5")
```

The various ways to configure the environment are described in detail in the article on Atari environments.
It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting.

| Environment | Valid Modes                     | Valid Difficulties | Default Mode |
|-------------|---------------------------------|--------------------|--------------|
| BankHeist   | `[0, 4, 8, 12, 16, 20, 24, 28]` | `[0, ..., 3]`      | `0`          |

You may use the suffix "-ram" to switch to the RAM observation space. In v0 and v4, the suffixes "Deterministic" and "NoFrameskip"
are available. These are no longer supported in v5. In order to obtain equivalent behavior, pass keyword arguments to `gymnasium.make` as outlined in
the general article on Atari environments.
The versions v0 and v4 are not contained in the "ALE" namespace. I.e. they are instantiated via `gymnasium.make("BankHeist-v0")`.

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the
general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The entire action space is used by default. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release (1.0.0)
