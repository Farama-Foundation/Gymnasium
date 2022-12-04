---
title: Defender
---

# Defender

```{figure} ../../_static/videos/atari/defender.gif
:width: 120px
:name: Defender
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                     |
|-------------------|-------------------------------------|
| Action Space      | Discrete(18)                        |
| Observation Space | (210, 160, 3)                       |
| Observation High  | 255                                 |
| Observation Low   | 0                                   |
| Import            | `gymnasium.make("ALE/Defender-v5")` |

## Description

Aliens attack the earth. You control a spaceship and must defend humanity by destroying alien ships and rescuing humanoids.
You have three lives and three smart bombs. You lose a live when you are shot down by an alien spaceship.
Points are scored by destroying enemies and retrieving humans that are being abducted. You have an unlimited number of
laser missiles.
Detailed documentation can be found on [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=128)

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

You receive points for destroying enemies, rescuing abducted humans and keeping humans alive.
For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=128).

## Arguments

```python
env = gymnasium.make("ALE/Defender-v5")
```

The various ways to configure the environment are described in detail in the article on Atari environments.
It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting.

| Environment | Valid Modes       | Valid Difficulties | Default Mode |
|-------------|-------------------|--------------------|--------------|
| Defender    | `[1, ..., 9, 16]` | `[0, 1]`           | `1`          |

You may use the suffix "-ram" to switch to the RAM observation space. In v0 and v4, the suffixes "Deterministic" and "NoFrameskip"
are available. These are no longer supported in v5. In order to obtain equivalent behavior, pass keyword arguments to `gymnasium.make` as outlined in
the general article on Atari environments.
The versions v0 and v4 are not contained in the "ALE" namespace. I.e. they are instantiated via `gymnasium.make("Defender-v0")`.

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the
general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The entire action space is used by default. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release (1.0.0)
