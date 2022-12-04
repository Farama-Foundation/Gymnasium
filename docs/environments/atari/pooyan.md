---
title: Pooyan
---

# Pooyan

```{figure} ../../_static/videos/atari/pooyan.gif
:width: 120px
:name: Pooyan
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                   |
|-------------------|-----------------------------------|
| Action Space      | Discrete(18)                      |
| Observation Space | (210, 160, 3)                     |
| Observation High  | 255                               |
| Observation Low   | 0                                 |
| Import            | `gymnasium.make("ALE/Pooyan-v5")` |

## Description

You are a mother pig protecting her piglets (Pooyans) from wolves. In the first scene, you can move up and down a rope. Try to shoot the worker's balloons, while guarding yourself from attacks. If the wolves reach the ground safely they will get behind and try to eat you. In the second scene, the wolves try to float up. You have to try and stop them using arrows and bait. You die if a wolf eats you, or a stone or rock hits you.
Detailed documentation can be found on [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=372)

## Actions

By default, all actions that can be performed on an Atari 2600 are available in this environment.
However, if you use v0 or v4 or specify `full_action_space=False` during initialization, only a reduced
number of actions (those that are meaningful in this game) are available. The reduced action space may depend on the flavor of the environment (the combination of `mode` and `difficulty`). The reduced action space for the default
flavor looks like this:

| Num | Action   |
|-----|----------|
| 0   | NOOP     |
| 1   | FIRE     |
| 2   | UP       |
| 3   | DOWN     |
| 4   | UPFIRE   |
| 5   | DOWNFIRE |

## Observations

By default, the environment returns the RGB image that is displayed to human players as an observation. However, it is possible to observe

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

If you hit a balloon, wolf or stone with an arrow you score points.
For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=372).

## Arguments

```python
env = gymnasium.make("ALE/Pooyan-v5")
```

The various ways to configure the environment are described in detail in the article on Atari environments.
It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting.

| Environment | Valid Modes        | Valid Difficulties | Default Mode |
|-------------|--------------------|--------------------|--------------|
| Pooyan      | `[10, 30, 50, 70]` | `[0]`              | `10`         |

You may use the suffix "-ram" to switch to the RAM observation space. In v0 and v4, the suffixes "Deterministic" and "NoFrameskip"
are available. These are no longer supported in v5. In order to obtain equivalent behavior, pass keyword arguments to `gymnasium.make` as outlined in
the general article on Atari environments.
The versions v0 and v4 are not contained in the "ALE" namespace. I.e. they are instantiated via `gymnasium.make("Pooyan-v0")`.

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the
general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The entire action space is used by default. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release (1.0.0)
