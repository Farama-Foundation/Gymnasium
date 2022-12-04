---
title: Seaquest
---

# Seaquest

```{figure} ../../_static/videos/atari/seaquest.gif
:width: 120px
:name: Seaquest
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                     |
|-------------------|-------------------------------------|
| Action Space      | Discrete(18)                        |
| Observation Space | (210, 160, 3)                       |
| Observation High  | 255                                 |
| Observation Low   | 0                                   |
| Import            | `gymnasium.make("ALE/Seaquest-v0")` |

## Description

You control a sub able to move in all directions and fire torpedoes.
The goal is to retrieve as many divers as you
can, while dodging and blasting enemy subs and killer sharks; points will be awarded accordingly.

The game begins with one sub and three waiting on the horizon. Each time you
increase your score by 10,000 points, an extra sub will be delivered to your
base.  You can only have six reserve subs on the screen at one time.

Your sub will explode if it collides with anything
except your own divers.

The sub has a limited amount of oxygen that decreases at a constant rate during the game. When the oxygen
tank is almost empty, you need to surface and if you don't do it in
time, yoursub will blow up and you'll lose one diver.  Each time you're forced
to surface, with less than six divers, you lose one diver as well.

Detailed documentation can be found on [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=424)

## Actions

By default, all actions that can be performed on an Atari 2600 are available in this environment.Even if you use v0 or v4 or specify `full_action_space=False` during initialization, all actions will be available in the default flavor.

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

## Arguments

```python
env = gymnasium.make("ALE/Seaquest-v5")
```

The various ways to configure the environment are described in detail in the article on Atari environments.
It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting.

| Environment | Valid Modes | Valid Difficulties | Default Mode |
|-------------|-------------|--------------------|--------------|
| Seaquest    | `[0]`       | `[0, 1]`           | `0`          |

You may use the suffix "-ram" to switch to the RAM observation space. In v0 and v4, the suffixes "Deterministic" and "NoFrameskip"
are available. These are no longer supported in v5. In order to obtain equivalent behavior, pass keyword arguments to `gymnasium.make` as outlined in
the general article on Atari environments.
The versions v0 and v4 are not contained in the "ALE" namespace. I.e. they are instantiated via `gymnasium.make("Seaquest-v0")`.

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the
general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The entire action space is used by default. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release (1.0.0)
