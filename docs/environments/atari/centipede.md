---
title: Centipede
---

# Centipede

```{figure} ../../_static/videos/atari/centipede.gif
:width: 120px
:name: Centipede
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                      |
|-------------------|--------------------------------------|
| Action Space      | Discrete(18)                         |
| Observation Space | (210, 160, 3)                        |
| Observation High  | 255                                  |
| Observation Low   | 0                                    |
| Import            | `gymnasium.make("ALE/Centipede-v5")` |

## Description

You are an elf and must use your magic wands to fend off spiders, fleas and centipedes. Your goal is to protect mushrooms in
an enchanted forest. If you are bitten by a spider, flea or centipede, you will be temporally paralyzed and you will
lose a magic wand. The game ends once you have lost all wands. You may receive additional wands after scoring
a sufficient number of points.
Detailed documentation can be found on [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=911).

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
 [255  ... 255]], (210, 160), uint8)
`

respectively. The general article on Atari environments outlines different ways to instantiate corresponding environments
via `gymnasium.make`.

### Rewards

You score points by hitting centipedes, scorpions, fleas and spiders. Additional points are awarded after every round
(i.e. after you have lost a wand) for mushrooms that were not destroyed.
Detailed documentation can be found on [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=911).

## Arguments

```python
env = gymnasium.make("ALE/Centipede-v5")
```

The various ways to configure the environment are described in detail in the article on Atari environments.
It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting.

| Environment | Valid Modes | Valid Difficulties | Default Mode |
|-------------|-------------|--------------------|--------------|
| Centipede   | `[22, 86]`  | `[0]`              | `22`         |

You may use the suffix "-ram" to switch to the RAM observation space. In v0 and v4, the suffixes "Deterministic" and "Noframeskip"
are available. These are no longer supported in v5. In order to obtain equivalent behavior, pass keyword arguments to `gymnasium.make` as outlined in
the general article on Atari environments.
The versions v0 and v4 are not contained in the "ALE" namespace. I.e. they are instantiated via `gymnasium.make("Centipede-v0")`.

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the
general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The entire action space is used by default. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release (1.0.0)
