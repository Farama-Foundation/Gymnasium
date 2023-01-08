---
title: Road Runner
---

# Road Runner

```{figure} ../../_static/videos/atari/road_runner.gif
:width: 120px
:name: RoadRunner
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                       |
|-------------------|---------------------------------------|
| Action Space      | Discrete(18)                          |
| Observation Space | (210, 160, 3)                         |
| Observation High  | 255                                   |
| Observation Low   | 0                                     |
| Import            | `gymnasium.make("ALE/RoadRunner-v0")` |

## Description

You control the Road Runner(TM) in a race; you can control the direction to run in and times to jumps.
The goal is to outrun Wile E. Coyote(TM) while avoiding the hazards of the desert.

The game begins with three lives.  You lose a life when the coyote
catches you, picks you up in a rocket, or shoots you with a cannon.  You also
lose a life when a truck hits you, you hit a land mine, you fall off a cliff,
or you get hit by a falling rock.

You score points (i.e. rewards) by eating seeds along the road, eating steel shot, and
destroying the coyote.

Detailed documentation can be found on [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=412)

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

Score points are your only reward. You get score points each time you:

| actions                                               | points |
|-------------------------------------------------------|--------|
| eat a pile of birdseed                                | 100    |
| eat steel shot                                        | 100    |
| get the coyote hit by a mine (cannonball, rock, etc.) | 200    |
| get the coyote hit by a truck                         | 1000   |

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=412).

## Arguments

```python
env = gymnasium.make("ALE/RoadRunner-v5")
```

The various ways to configure the environment are described in detail in the article on Atari environments.
It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting.

| Environment | Valid Modes | Valid Difficulties | Default Mode |
|-------------|-------------|--------------------|--------------|
| RoadRunner  | `[0]`       | `[0]`              | `0`          |

You may use the suffix "-ram" to switch to the RAM observation space. In v0 and v4, the suffixes "Deterministic" and "NoFrameskip"
are available. These are no longer supported in v5. In order to obtain equivalent behavior, pass keyword arguments to `gymnasium.make` as outlined in
the general article on Atari environments.
The versions v0 and v4 are not contained in the "ALE" namespace. I.e. they are instantiated via `gymnasium.make("RoadRunner-v0")`.

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the
general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The entire action space is used by default. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release (1.0.0)
