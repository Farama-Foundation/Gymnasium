---
firstpage:
lastpage:
---

# Atari

A set of Atari 2600 environments simulated through Stella and the Arcade Learning Environment.

```{toctree}
:hidden:
atari/adventure
atari/air_raid
atari/alien
atari/amidar
atari/assault
atari/asterix
atari/asteroids
atari/atlantis
atari/bank_heist
atari/battle_zone
atari/beam_rider
atari/berzerk
atari/bowling
atari/boxing
atari/breakout
atari/carnival
atari/centipede
atari/chopper_command
atari/crazy_climber
atari/defender
atari/demon_attack
atari/double_dunk
atari/elevator_action
atari/enduro
atari/fishing_derby
atari/freeway
atari/frostbite
atari/gopher
atari/gravitar
atari/hero
atari/ice_hockey
atari/jamesbond
atari/journey_escape
atari/kangaroo
atari/krull
atari/kung_fu_master
atari/montezuma_revenge
atari/ms_pacman
atari/name_this_game
atari/phoenix
atari/pitfall
atari/pong
atari/pooyan
atari/private_eye
atari/qbert
atari/riverraid
atari/road_runner
atari/robotank
atari/seaquest
atari/skiing
atari/solaris
atari/space_invaders
atari/star_gunner
atari/tennis
atari/time_pilot
atari/tutankham
atari/up_n_down
atari/venture
atari/video_pinball
atari/wizard_of_wor
atari/yars_revenge
atari/zaxxon
```

```{raw} html
   :file: atari/list.html
```

Atari environments are simulated via the Arcade Learning Environment (ALE) [[1]](#1).

## AutoROM (installing the ROMs)

ALE-py doesn't include the atari ROMs (`pip install gymnasium[atari]`) which are necessary to make any of the atari environments.
To install the atari ROM, use `pip install gymnasium[accept-rom-license]` which will install AutoROM and download the ROMs, install them in the default location.
In doing so, you agree to TODO

It is possible to install the ROMs in an alternative location, [repo](https://github.com/Farama-Foundation/AutoROM) has more information.

## Action Space

The action space is a subset of the following discrete set of legal actions:

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

The reduced action space of an Atari environment may depend on the "flavor" of the game. You can specify the flavor by providing
the arguments `difficulty` and `mode` when constructing the environment. This documentation only provides details on the
action spaces of default flavor choices.

## Observation Space

The observation issued by an Atari environment may be:

- the RGB image that is displayed to a human player,
- a grayscale version of that image or
- the state of the 128 Bytes of RAM of the console.

## Rewards

The exact reward dynamics depend on the environment and are usually documented in the game's manual. You can
find these manuals on [AtariAge](https://atariage.com/).

## Stochasticity

It was pointed out in [[1]](#1) that Atari games are entirely deterministic. Thus, agents could achieve
state-of-the-art performance by simply memorizing an optimal sequence of actions while completely ignoring observations from the environment.
To avoid this, ALE implements sticky actions: Instead of always simulating the action passed to the environment, there is a small
probability that the previously executed action is used instead.

On top of this, Gymnasium implements stochastic frame skipping: In each environment step, the action is repeated for a random
number of frames. This behavior may be altered by setting the keyword argument `frameskip` to either a positive integer or
a tuple of two positive integers. If `frameskip` is an integer, frame skipping is deterministic, and in each step the action is
repeated `frameskip` many times. Otherwise, if `frameskip` is a tuple, the number of skipped frames is chosen uniformly at
random between `frameskip[0]` (inclusive) and `frameskip[1]` (exclusive) in each environment step.

## Common Arguments

When initializing Atari environments via `gymnasium.make`, you may pass some additional arguments. These work for any
Atari environment. However, legal values for `mode` and `difficulty` depend on the environment.

- **mode**: `int`. Game mode, see [[2]](#2). Legal values depend on the environment and are listed in the table above.

- **difficulty**: `int`. The difficulty of the game, see [[2]](#2). Legal values depend on the environment and are listed in
the table above. Together with `mode`, this determines the "flavor" of the game.

- **obs_type**: `str`. This argument determines what observations are returned by the environment. Its values are:
	- ram: The 128 Bytes of RAM are returned
	- rgb: An RGB rendering of the game is returned
	- grayscale: A grayscale rendering is returned

- **frameskip**: `int` or a tuple of two `int`s. This argument controls stochastic frame skipping, as described in the section on stochasticity.

- **repeat_action_probability**: `float`. The probability that an action sticks, as described in the section on stochasticity.

- **full_action_space**: `bool`. If set to `True`, the action space consists of all legal actions on the console. Otherwise, the
action space will be reduced to a subset.

- **render_mode**: `str`. Specifies the rendering mode. Its values are:
	- human: We'll interactively display the screen and enable game sounds. This will lock emulation to the ROMs specified FPS
	- rgb_array: we'll return the `rgb` key in step metadata with the current environment RGB frame.
> It is highly recommended to specify `render_mode` during construction instead of calling `env.render()`.
> This will guarantee proper scaling, audio support, and proper framerates

## Version History and Naming Schemes

All Atari games are available in three versions. They differ in the default settings of the arguments above.
The differences are listed in the following table:

| Version | `frameskip=` | `repeat_action_probability=` | `full_action_space=` |
|---------|--------------|------------------------------|----------------------|
| v0      | `(2, 5,)`    | `0.25`                       | `False`              |
| v4      | `(2, 5,)`    | `0.0`                        | `False`              |
| v5      | `5`          | `0.25`                       | `True`               |

> Version v5 follows the best practices outlined in [[2]](#2). Thus, it is recommended to transition to v5 and
> customize the environment using the arguments above, if necessary.

For each Atari game, several different configurations are registered in Gymnasium. The naming schemes are analogous for
v0 and v4. Let us take a look at all variations of Amidar-v0 that are registered with gymnasium:

| Name                       | `obs_type=` | `frameskip=` | `repeat_action_probability=` | `full_action_space=` |
|----------------------------|-------------|--------------|------------------------------|----------------------|
| Amidar-v0                  | `"rgb"`     | `(2, 5,)`    | `0.25`                       | `False`              |
| AmidarDeterministic-v0     | `"rgb"`     | `4`          | `0.0`                        | `False`              |
| AmidarNoframeskip-v0       | `"rgb"`     | `1`          | `0.25`                       | `False`              |
| Amidar-ram-v0              | `"ram"`     | `(2, 5,)`    | `0.25`                       | `False`              |
| Amidar-ramDeterministic-v0 | `"ram"`     | `4`          | `0.0`                        | `False`              |
| Amidar-ramNoframeskip-v0   | `"ram"`     | `1`          | `0.25`                       | `False`              |

Things change in v5: The suffixes "Deterministic" and "NoFrameskip" are no longer available. Instead, you must specify the
environment configuration via arguments passed to `gymnasium.make`. Moreover, the v5 environments
are in the "ALE" namespace. The suffix "-ram" is still available. Thus, we get the following table:

| Name              | `obs_type=` | `frameskip=` | `repeat_action_probability=` | `full_action_space=` |
|-------------------|-------------|--------------|------------------------------|----------------------|
| ALE/Amidar-v5     | `"rgb"`     | `5`          | `0.25`                       | `True`               |
| ALE/Amidar-ram-v5 | `"ram"`     | `5`          | `0.25`                       | `True`               |

## Flavors

Some games allow the user to set a difficulty level and a game mode. Different modes/difficulties may have different
game dynamics and (if a reduced action space is used) different action spaces. We follow the convention of [[2]](#2) and
refer to the combination of difficulty level and game mode as a flavor of a game. The following table shows
the available modes and difficulty levels for different Atari games:

| Environment      | Valid Modes                                     | Default Mode   |
|------------------|-------------------------------------------------|----------------|
| Adventure        | `[0, 1, 2]`                                     | `0`            |
| AirRaid          | `[1, ..., 8]`                                   | `1`            |
| Alien            | `[0, ..., 3]`                                   | `0`            |
| Amidar           | `[0]`                                           | `0`            |
| Assault          | `[0]`                                           | `0`            |
| Asterix          | `[0]`                                           | `0`            |
| Asteroids        | `[0, ..., 31, 128]`                             | `0`            |
| Atlantis         | `[0, ..., 3]`                                   | `0`            |
| BankHeist        | `[0, 4, 8, 12, 16, 20, 24, 28]`                 | `0`            |
| BattleZone       | `[1, 2, 3]`                                     | `1`            |
| BeamRider        | `[0]`                                           | `0`            |
| Berzerk          | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18]`       | `1`            |
| Bowling          | `[0, 2, 4]`                                     | `0`            |
| Boxing           | `[0]`                                           | `0`            |
| Breakout         | `[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]` | `0`            |
| Carnival         | `[0]`                                           | `0`            |
| Centipede        | `[22, 86]`                                      | `22`           |
| ChopperCommand   | `[0, 2]`                                        | `0`            |
| CrazyClimber     | `[0, ..., 3]`                                   | `0`            |
| Defender         | `[1, ..., 9, 16]`                               | `1`            |
| DemonAttack      | `[1, 3, 5, 7]`                                  | `1`            |
| DoubleDunk       | `[0, ..., 15]`                                  | `0`            |
| ElevatorAction   | `[0]`                                           | `0`            |
| Enduro           | `[0]`                                           | `0`            |
| FishingDerby     | `[0]`                                           | `0`            |
| Freeway          | `[0, ..., 7]`                                   | `0`            |
| Frostbite        | `[0, 2]`                                        | `0`            |
| Gopher           | `[0, 2]`                                        | `0`            |
| Gravitar         | `[0, ..., 4]`                                   | `0`            |
| Hero             | `[0, ..., 4]`                                   | `0`            |
| IceHockey        | `[0, 2]`                                        | `0`            |
| Jamesbond        | `[0, 1]`                                        | `0`            |
| JourneyEscape    | `[0]`                                           | `0`            |
| Kangaroo         | `[0, 1]`                                        | `0`            |
| Krull            | `[0]`                                           | `0`            |
| KungFuMaster     | `[0]`                                           | `0`            |
| MontezumaRevenge | `[0]`                                           | `0`            |
| MsPacman         | `[0, ..., 3]`                                   | `0`            |
| NameThisGame     | `[8, 24, 40]`                                   | `8`            |
| Phoenix          | `[0]`                                           | `0`            |
| Pitfall          | `[0]`                                           | `0`            |
| Pong             | `[0, 1]`                                        | `0`            |
| Pooyan           | `[10, 30, 50, 70]`                              | `10`           |
| PrivateEye       | `[0, ..., 4]`                                   | `0`            |
| Qbert            | `[0]`                                           | `0`            |
| Riverraid        | `[0]`                                           | `0`            |
| RoadRunner       | `[0]`                                           | `0`            |
| Robotank         | `[0]`                                           | `0`            |
| Seaquest         | `[0]`                                           | `0`            |
| Skiing           | `[0]`                                           | `0`            |
| Solaris          | `[0]`                                           | `0`            |
| SpaceInvaders    | `[0, ..., 15]`                                  | `0`            |
| StarGunner       | `[0, ..., 3]`                                   | `0`            |
| Tennis           | `[0, 2]`                                        | `0`            |
| TimePilot        | `[0]`                                           | `0`            |
| Tutankham        | `[0, 4, 8, 12]`                                 | `0`            |
| UpNDown          | `[0]`                                           | `0`            |
| Venture          | `[0]`                                           | `0`            |
| VideoPinball     | `[0, 2]`                                        | `0`            |
| WizardOfWor      | `[0]`                                           | `0`            |
| YarsRevenge      | `[0, 32, 64, 96]`                               | `0`            |
| Zaxxon           | `[0, 8, 16, 24]`                                | `0`            |

> Each game also has a valid difficulty for the opposing AI, which has a different range depending on the game. These values can have a range of 0 - n, where n can be found at [the ALE documentation](https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/docs/games.md)

## References

(#1)=
<a id="1">[1]</a>
MG Bellemare, Y Naddaf, J Veness, and M Bowling.
"The arcade learning environment: An evaluation platform for general agents."
Journal of Artificial Intelligence Research (2012).

(#2)=
<a id="2">[2]</a>
Machado et al.
"Revisiting the Arcade Learning Environment: Evaluation Protocols
and Open Problems for General Agents"
Journal of Artificial Intelligence Research (2018)
URL: https://jair.org/index.php/jair/article/view/11182
