---
title: Gravitar
---

# Gravitar

```{figure} ../../_static/videos/atari/gravitar.gif
:width: 120px
:name: Gravitar
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                     |
|-------------------|-------------------------------------|
| Action Space      | Discrete(18)                        |
| Observation Space | (210, 160, 3)                       |
| Observation High  | 255                                 |
| Observation Low   | 0                                   |
| Import            | `gymnasium.make("ALE/Gravitar-v5")` |

## Description

The player controls a small blue spacecraft. The game starts in a fictional solar system with several planets to explore. If the player moves his ship into a planet, he will be taken to a side-view landscape. Player has to destroy red bunkers [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=223).

### Rewards

The exact reward dynamics depend on the environment and are usually documented in the game's manual. You can
find these manuals on [AtariAge](https://atariage.com/manual_html_page.php?SoftwareLabelID=223).

Atari environments are simulated via the Arcade Learning Environment (ALE) [[1]](#1).

### Action Space

The action space a subset of the following discrete set of legal actions:

| Num | Action        |
|-----|---------------|
| 0   | NOOP          |
| 1   | FIRE          |
| 2   | UP            |
| 3   | RIGHT         |
| 4   | LEFT          |
| 5   | DOWN          |
| 6   | UPRIGHT       |
| 7   | UPLEFT        |
| 8   | DOWNRIGHT     |
| 9   | DOWNLEFT      |
| 10  | UPFIRE        |
| 11  | RIGHTFIRE     |
| 12  | LEFTFIRE      |
| 13  | DOWNFIRE      |
| 14  | UPRIGHTFIRE   |
| 15  | UPLEFTFIRE    |
| 16  | DOWNRIGHTFIRE |
| 17  | DOWNLEFTFIRE  |

If you use v0 or v4 and the environment is initialized via `make`, the action space will usually be much smaller since most legal actions don't have
any effect. Thus, the enumeration of the actions will differ. The action space can be expanded to the full
legal space by passing the keyword argument `full_action_space=True` to `make`.

The reduced action space of an Atari environment may depend on the flavor of the game. You can specify the flavor by providing
the arguments `difficulty` and `mode` when constructing the environment. This documentation only provides details on the
action spaces of default flavors.

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

## Arguments

```python
env = gymnasium.make("ALE/Gravitar-v5")
```

The various ways to configure the environment are described in detail in the article on Atari environments.

| Environment | Valid Modes   | Valid Difficulties | Default Mode |
|-------------|---------------|--------------------|--------------|
| Gravitar    | `[0, ..., 4]` | `[0]`              | `0`          |

You may use the suffix "-ram" to switch to the RAM observation space. In v0 and v4, the suffixes "Deterministic" and "NoFrameskip"
are available. These are no longer supported in v5. In order to obtain equivalent behavior, pass keyword arguments to `gymnasium.make` as outlined in
the general article on Atari environments.
The versions v0 and v4 are not contained in the "ALE" namespace. I.e. they are instantiated via `gymnasium.make("Gravitar-v0")`.

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the
general article on Atari environments.

| Version | `frameskip=` | `repeat_action_probability=` | `full_action_space=` |
|---------|--------------|------------------------------|----------------------|
| v0      | `(2, 5,)`    | `0.25`                       | `False`              |
| v4      | `(2, 5,)`    | `0.0`                        | `False`              |
| v5      | `5`          | `0.25`                       | `True`               |

> Version v5 follows the best practices outlined in [[2]](#2). Thus, it is recommended to transition to v5 and
> customize the environment using the arguments above, if necessary.
