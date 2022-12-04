---
title: JourneyEscape
---

# JourneyEscape

```{figure} ../../_static/videos/atari/journey_escape.gif
:width: 120px
:name: JourneyEscape
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                          |
|-------------------|------------------------------------------|
| Action Space      | Discrete(18)                             |
| Observation Space | (210, 160, 3)                            |
| Observation High  | 255                                      |
| Observation Low   | 0                                        |
| Import            | `gymnasium.make("ALE/JourneyEscape-v5")` |

## Description

You must lead all 5 members of JOURNEY through waves of pesky characters and backstage obstacles to the Scarab Escape Vehicle before time runs out.
You must also protect $50,000 in concert cash from grasping groupies, photographers, and promoters.
Detailed documentation can be found on [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=252)

## Actions

By default, all actions that can be performed on an Atari 2600 are available in this environment.
However, if you use v0 or v4 or specify `full_action_space=False` during initialization, only a reduced
number of actions (those that are meaningful in this game) are available. The reduced action space may depend
on the flavor of the environment (the combination of `mode` and `difficulty`). The reduced action space for the default
flavor looks like this:

| Num | Action        |
|-----|---------------|
| 0   | NOOP          |
| 2   | UP            |
| 3   | RIGHT         |
| 4   | LEFT          |
| 5   | DOWN          |
| 6   | UPRIGHT       |
| 7   | UPLEFT        |
| 8   | DOWNRIGHT     |
| 9   | DOWNLEFT      |
| 11  | RIGHTFIRE     |
| 12  | LEFTFIRE      |
| 13  | DOWNFIRE      |
| 14  | UPRIGHTFIRE   |
| 15  | UPLEFTFIRE    |
| 16  | DOWNRIGHTFIRE |
| 17  | DOWNLEFTFIRE  |

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

respectively. [The general article on Atari environments](https://brosa.ca/blog/ale-release-v0.7) outlines different ways to instantiate corresponding environments
via `gymnasium.make`.

### Rewards

At the start of the game, you will have $50,000 and 60 units of time.
Your end game score with be dependent on how much time you have remaining and who you encounter along the way.
For a more detailed documentation, consult [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=252).

## Arguments

```python
env = gymnasium.make("ALE/JourneyEscape-v5")
```

The various ways to configure the environment are described in detail in the article on Atari environments.
It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting.

|      Environment | Valid Modes                                                                                                                                                                         | Valid Difficulties | Default Mode |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|--------------|
|    JourneyEscape | `[0]`                                                                                                                                                                               |           `[0, 1]` | `0`          |

You may use the suffix "-ram" to switch to the RAM observation space. In v0 and v4, the suffixes "Deterministic" and "NoFrameskip"
are available. These are no longer supported in v5. In order to obtain equivalent behavior, pass keyword arguments to `gymnasium.make` as outlined in
the general article on Atari environments.
The versions v0 and v4 are not contained in the "ALE" namespace. I.e. they are instantiated via `gymnasium.make("JourneyEscape-v0")`.

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the
general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The entire action space is used by default. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release (1.0.0)
