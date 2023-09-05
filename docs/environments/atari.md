---
firstpage:
lastpage:
---

# Atari

A set of Atari 2600 environments simulated through [Stella](https://github.com/stella-emu/stella) and the [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment).

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
atari/atlantis2
atari/backgammon
atari/bank_heist
atari/basic_math
atari/battle_zone
atari/beam_rider
atari/berzerk
atari/blackjack
atari/bowling
atari/boxing
atari/breakout
atari/carnival
atari/casino
atari/centipede
atari/chopper_command
atari/crazy_climber
atari/crossbow
atari/darkchambers
atari/defender
atari/demon_attack
atari/donkey_kong
atari/double_dunk
atari/earthworld
atari/elevator_action
atari/enduro
atari/entombed
atari/et
atari/fishing_derby
atari/flag_capture
atari/freeway
atari/frogger
atari/frostbite
atari/galaxian
atari/gopher
atari/gravitar
atari/hangman
atari/haunted_house
atari/hero
atari/human_cannonball
atari/ice_hockey
atari/jamesbond
atari/journey_escape
atari/kaboom
atari/kangaroo
atari/keystone_kapers
atari/king_kong
atari/klax
atari/koolaid
atari/krull
atari/kung_fu_master
atari/laser_gates
atari/lost_luggage
atari/mario_bros
atari/miniature_golf
atari/montezuma_revenge
atari/mr_do
atari/ms_pacman
atari/name_this_game
atari/othello
atari/pacman
atari/phoenix
atari/pitfall
atari/pitfall2
atari/pong
atari/pooyan
atari/private_eye
atari/qbert
atari/riverraid
atari/road_runner
atari/robotank
atari/seaquest
atari/sir_lancelot
atari/skiing
atari/solaris
atari/space_invaders
atari/space_war
atari/star_gunner
atari/superman
atari/surround
atari/tennis
atari/tetris
atari/tic_tac_toe_3d
atari/time_pilot
atari/trondead
atari/turmoil
atari/tutankham
atari/up_n_down
atari/venture
atari/video_checkers
atari/video_chess
atari/video_cube
atari/video_pinball
atari/wizard_of_wor
atari/word_zapper
atari/yars_revenge
atari/zaxxon
```

```{raw} html
   :file: atari/list.html
```

Atari environments are simulated via the Arcade Learning Environment (ALE) [[1]](#1) through the Stella emulator.

## AutoROM (installing the ROMs)

ALE-py doesn't include the atari ROMs (`pip install gymnasium[atari]`) which are necessary to make any of the atari environments.
To install the atari ROM, use `pip install gymnasium[accept-rom-license]` which will install AutoROM and download the ROMs, install them in the default location.
In doing so, you agree to own a license to these Atari 2600 ROMs and agree to not distribution these ROMS.

It is possible to install the ROMs in an alternative location, [AutoROM](https://github.com/Farama-Foundation/AutoROM) has more information.

## Action Space

Each environment will use a sub-set of the full action space listed below:

| Value   | Meaning      | Value   | Meaning         | Value   | Meaning        |
|---------|--------------|---------|-----------------|---------|----------------|
| `0`     | `NOOP`       | `1`     | `FIRE`          | `2`     | `UP`           |
| `3`     | `RIGHT`      | `4`     | `LEFT`          | `5`     | `DOWN`         |
| `6`     | `UPRIGHT`    | `7`     | `UPLEFT`        | `8`     | `DOWNRIGHT`    |
| `9`     | `DOWNLEFT`   | `10`    | `UPFIRE`        | `11`    | `RIGHTFIRE`    |
| `12`    | `LEFTFIRE`   | `13`    | `DOWNFIRE`      | `14`    | `UPRIGHTFIRE`  |
| `15`    | `UPLEFTFIRE` | `16`    | `DOWNRIGHTFIRE` | `17`    | `DOWNLEFTFIRE` |



By default, most environments use a smaller subset of the legal actions excluding any actions that don't have an effect in the game.
If users are interested in using all possible actions, pass the keyword argument `full_action_space=True` to `gymnasium.make`.

## Observation Space

The Atari environments observation can be
1. The RGB image that is displayed to a human player using `obs_type="rgb"` with observation space `Box(0, 255, (210, 160, 3), np.uint8)`
2. The grayscale version of the RGB image using `obs_type="grayscale"` with observation space `Box(0, 255, (210, 160), np.uint8)`
3. The RAM state (128 bytes) from the console using `obs_type="ram"` with observation space `Box(0, 255, (128), np.uint8)`

## Rewards

The exact reward dynamics depend on the environment and are usually documented in the game's manual. You can
find these manuals on [AtariAge](https://atariage.com/).

## Stochasticity

As the Atari games are entirely deterministic, agents could achieve
state-of-the-art performance by simply memorizing an optimal sequence of actions while completely ignoring observations from the environment.

To avoid this, there are several methods to avoid this.

1. Sticky actions: Instead of always simulating the action passed to the environment, there is a small
probability that the previously executed action is used instead. In the v0 and v5 environments, the probability of
repeating an action is `25%` while in v4 environments, the probability is `0%`. Users can specify the repeat action
probability using `repeat_action_probability` to `make`.
2. Frameskipping: On each environment step, the action can be repeated for a random number of frames. This behavior
may be altered by setting the keyword argument `frameskip` to either a positive integer or
a tuple of two positive integers. If `frameskip` is an integer, frame skipping is deterministic, and in each step the action is
repeated `frameskip` many times. Otherwise, if `frameskip` is a tuple, the number of skipped frames is chosen uniformly at
random between `frameskip[0]` (inclusive) and `frameskip[1]` (exclusive) in each environment step.

## Common Arguments

When initializing Atari environments via `gymnasium.make`, you may pass some additional arguments. These work for any
Atari environment.

- **mode**: `int`. Game mode, see [[2]](#2). Legal values depend on the environment and are listed in the table above.

- **difficulty**: `int`. The difficulty of the game, see [[2]](#2). Legal values depend on the environment and are listed in
the table above. Together with `mode`, this determines the "flavor" of the game.

- **obs_type**: `str`. This argument determines what observations are returned by the environment. Its values are:
	- "ram": The 128 Bytes of RAM are returned
	- "rgb": An RGB rendering of the game is returned
	- "grayscale": A grayscale rendering is returned

- **frameskip**: `int` or a tuple of two `int`s. This argument controls stochastic frame skipping, as described in the section on stochasticity.

- **repeat_action_probability**: `float`. The probability that an action is repeated, also called "sticky actions", as described in the section on stochasticity.

- **full_action_space**: `bool`. If set to `True`, the action space consists of all legal actions on the console. Otherwise, the
action space will be reduced to a subset.

- **render_mode**: `str`. Specifies the rendering mode. Its values are:
	- human: Display the screen and enable game sounds. This will lock emulation to the ROMs specified FPS
	- rgb_array: Returns the current environment RGB frame of the environment.

## Version History and Naming Schemes

All Atari games are available in three versions. They differ in the default settings of the arguments above.
The differences are listed in the following table:

| Version | `frameskip=`                        | `repeat_action_probability=` | `full_action_space=` |
|---------|-------------------------------------|------------------------------|----------------------|
| v0      | Varies with the suffix (see below). | `0.25`                       | `False`              |
| v4      | Varies with the suffix (see below). | `0.0`                        | `False`              |
| v5      | `4`                                 | `0.25`                       | `False`              |

> Version v5 follows the best practices outlined in [[2]](#2). Thus, it is recommended to transition to v5 and
customize the environment using the arguments above, if necessary.

For each Atari game, several different configurations are registered in Gymnasium. The naming schemes are analogous for
v0 and v4. Let us take a look at all variations of Amidar-v0 that are registered with gymnasium:

| Name                       | `obs_type=` | `frameskip=` | `repeat_action_probability=` |
|----------------------------|-------------|--------------|------------------------------|
| Amidar-v0                  | `"rgb"`     | `(2, 5,)`    | `0.25`                       |
| AmidarDeterministic-v0     | `"rgb"`     | `4`          | `0.0`                        |
| AmidarNoframeskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Amidar-ram-v0              | `"ram"`     | `(2, 5,)`    | `0.25`                       |
| Amidar-ramDeterministic-v0 | `"ram"`     | `4`          | `0.0`                        |
| Amidar-ramNoframeskip-v0   | `"ram"`     | `1`          | `0.25`                       |

Things change in v5: The suffixes "Deterministic" and "NoFrameskip" are no longer available. Instead, you must specify the
environment configuration via arguments passed to `gymnasium.make`. Moreover, the v5 environments
are in the "ALE" namespace. The suffix "-ram" is still available. Thus, we get the following table:

| Name              | `obs_type=` | `frameskip=` | `repeat_action_probability=` |
|-------------------|-------------|--------------|------------------------------|
| ALE/Amidar-v5     | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Amidar-ram-v5 | `"ram"`     | `4`          | `0.25`                       |

## Flavors

Some games allow the user to set a difficulty level and a game mode. Different modes/difficulties may have different
game dynamics and (if a reduced action space is used) different action spaces. We follow the convention of [[2]](#2) and
refer to the combination of difficulty level and game mode as a flavor of a game. The following table shows
the available modes and difficulty levels for different Atari games:

| Environment      | Possible Modes                                  |   Default Mode | Possible Difficulties   |   Default Difficulty |
|------------------|-------------------------------------------------|----------------|-------------------------|----------------------|
| Adventure        | [0, 1, 2]                                       |              0 | [0, 1, 2, 3]            |                    0 |
| AirRaid          | [1, ..., 8]                                     |              1 | [0]                     |                    0 |
| Alien            | [0, 1, 2, 3]                                    |              0 | [0, 1, 2, 3]            |                    0 |
| Amidar           | [0]                                             |              0 | [0, 3]                  |                    0 |
| Assault          | [0]                                             |              0 | [0]                     |                    0 |
| Asterix          | [0]                                             |              0 | [0]                     |                    0 |
| Asteroids        | [0, ..., 31, 128]                               |              0 | [0, 3]                  |                    0 |
| Atlantis         | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| Atlantis2        | [0]                                             |              0 | [0]                     |                    0 |
| Backgammon       | [0]                                             |              0 | [3]                     |                    0 |
| BankHeist        | [0, 4, 8, 12, 16, 20, 24, 28]                   |              0 | [0, 1, 2, 3]            |                    0 |
| BasicMath        | [5, 6, 7, 8]                                    |              5 | [0, 2, 3]               |                    0 |
| BattleZone       | [1, 2, 3]                                       |              1 | [0]                     |                    0 |
| BeamRider        | [0]                                             |              0 | [0, 1]                  |                    0 |
| Berzerk          | [1, ..., 9, 16, 17, 18]                         |              1 | [0]                     |                    0 |
| Blackjack        | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| Bowling          | [0, 2, 4]                                       |              0 | [0, 1]                  |                    0 |
| Boxing           | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| Breakout         | [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]   |              0 | [0, 1]                  |                    0 |
| Carnival         | [0]                                             |              0 | [0]                     |                    0 |
| Casino           | [0, 2, 3]                                       |              0 | [0, 1, 2, 3]            |                    0 |
| Centipede        | [22, 86]                                        |             22 | [0]                     |                    0 |
| ChopperCommand   | [0, 2]                                          |              0 | [0, 1]                  |                    0 |
| CrazyClimber     | [0, 1, 2, 3]                                    |              0 | [0, 1]                  |                    0 |
| Crossbow         | [0, 2, 4, 6]                                    |              0 | [0, 1]                  |                    0 |
| Darkchambers     | [0]                                             |              0 | [0]                     |                    0 |
| Defender         | [1, ..., 9, 16]                                 |              1 | [0, 1]                  |                    0 |
| DemonAttack      | [1, 3, 5, 7]                                    |              1 | [0, 1]                  |                    0 |
| DonkeyKong       | [0]                                             |              0 | [0]                     |                    0 |
| DoubleDunk       | [0, ..., 15]                                    |              0 | [0]                     |                    0 |
| Earthworld       | [0]                                             |              0 | [0]                     |                    0 |
| ElevatorAction   | [0]                                             |              0 | [0]                     |                    0 |
| Enduro           | [0]                                             |              0 | [0]                     |                    0 |
| Entombed         | [0]                                             |              0 | [0, 2]                  |                    0 |
| Et               | [0, 1, 2]                                       |              0 | [0, 1, 2, 3]            |                    0 |
| FishingDerby     | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| FlagCapture      | [8, 9, 10]                                      |              8 | [0]                     |                    0 |
| Freeway          | [0, ..., 7]                                     |              0 | [0, 1]                  |                    0 |
| Frogger          | [0, 1, 2]                                       |              0 | [0, 1]                  |                    0 |
| Frostbite        | [0, 2]                                          |              0 | [0]                     |                    0 |
| Galaxian         | [1, ..., 9]                                     |              1 | [0, 1]                  |                    0 |
| Gopher           | [0, 2]                                          |              0 | [0, 1]                  |                    0 |
| Gravitar         | [0, 1, 2, 3, 4]                                 |              0 | [0]                     |                    0 |
| Hangman          | [0, 1, 2, 3]                                    |              0 | [0, 1]                  |                    0 |
| HauntedHouse     | [0, ..., 8]                                     |              0 | [0, 1]                  |                    0 |
| Hero             | [0, 1, 2, 3, 4]                                 |              0 | [0]                     |                    0 |
| HumanCannonball  | [0, ..., 7]                                     |              0 | [0, 1]                  |                    0 |
| IceHockey        | [0, 2]                                          |              0 | [0, 1, 2, 3]            |                    0 |
| Jamesbond        | [0, 1]                                          |              0 | [0]                     |                    0 |
| JourneyEscape    | [0]                                             |              0 | [0, 1]                  |                    0 |
| Kaboom           | [0]                                             |              0 | [0]                     |                    0 |
| Kangaroo         | [0, 1]                                          |              0 | [0]                     |                    0 |
| KeystoneKapers   | [0]                                             |              0 | [0]                     |                    0 |
| KingKong         | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| Klax             | [0, 1, 2]                                       |              0 | [0]                     |                    0 |
| Koolaid          | [0]                                             |              0 | [0]                     |                    0 |
| Krull            | [0]                                             |              0 | [0]                     |                    0 |
| KungFuMaster     | [0]                                             |              0 | [0]                     |                    0 |
| LaserGates       | [0]                                             |              0 | [0]                     |                    0 |
| LostLuggage      | [0, 1]                                          |              0 | [0, 1]                  |                    0 |
| MarioBros        | [0, 2, 4, 6]                                    |              0 | [0]                     |                    0 |
| MiniatureGolf    | [0]                                             |              0 | [0, 1]                  |                    0 |
| MontezumaRevenge | [0]                                             |              0 | [0]                     |                    0 |
| MrDo             | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| MsPacman         | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| NameThisGame     | [8, 24, 40]                                     |              8 | [0, 1]                  |                    0 |
| Othello          | [0, 1, 2]                                       |              0 | [0, 2]                  |                    0 |
| Pacman           | [0, ..., 7]                                     |              0 | [0, 1]                  |                    0 |
| Phoenix          | [0]                                             |              0 | [0]                     |                    0 |
| Pitfall          | [0]                                             |              0 | [0]                     |                    0 |
| Pitfall2         | [0]                                             |              0 | [0]                     |                    0 |
| Pong             | [0, 1]                                          |              0 | [0, 1, 2, 3]            |                    0 |
| Pooyan           | [10, 30, 50, 70]                                |             10 | [0]                     |                    0 |
| PrivateEye       | [0, 1, 2, 3, 4]                                 |              0 | [0, 1, 2, 3]            |                    0 |
| Qbert            | [0]                                             |              0 | [0, 1]                  |                    0 |
| Riverraid        | [0]                                             |              0 | [0, 1]                  |                    0 |
| RoadRunner       | [0]                                             |              0 | [0]                     |                    0 |
| Robotank         | [0]                                             |              0 | [0]                     |                    0 |
| Seaquest         | [0]                                             |              0 | [0, 1]                  |                    0 |
| SirLancelot      | [0]                                             |              0 | [0]                     |                    0 |
| Skiing           | [0]                                             |              0 | [0]                     |                    0 |
| Solaris          | [0]                                             |              0 | [0]                     |                    0 |
| SpaceInvaders    | [0, ..., 15]                                    |              0 | [0, 1]                  |                    0 |
| SpaceWar         | [6, ..., 17]                                    |              6 | [0]                     |                    0 |
| StarGunner       | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| Superman         | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| Surround         | [0, 2]                                          |              0 | [0, 1, 2, 3]            |                    0 |
| Tennis           | [0, 2]                                          |              0 | [0, 1, 2, 3]            |                    0 |
| Tetris           | [0]                                             |              0 | [0]                     |                    0 |
| TicTacToe3D      | [0, ..., 8]                                     |              0 | [0, 2]                  |                    0 |
| TimePilot        | [0]                                             |              0 | [0, 1, 2]               |                    0 |
| Trondead         | [0]                                             |              0 | [0, 1]                  |                    0 |
| Turmoil          | [0, ..., 8]                                     |              0 | [0]                     |                    0 |
| Tutankham        | [0, 4, 8, 12]                                   |              0 | [0]                     |                    0 |
| UpNDown          | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| Venture          | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| VideoCheckers    | [1, ..., 9, 11, ..., 19]                        |              1 | [0]                     |                    0 |
| VideoChess       | [0, 1, 2, 3, 4]                                 |              0 | [0]                     |                    0 |
| VideoCube        | [0, 1, 2, 100, 101, 102, ..., 5000, 5001, 5002] |              0 | [0, 1]                  |                    0 |
| VideoPinball     | [0, 2]                                          |              0 | [0, 1]                  |                    0 |
| WizardOfWor      | [0]                                             |              0 | [0, 1]                  |                    0 |
| WordZapper       | [0, ..., 23]                                    |              0 | [0, 1, 2, 3]            |                    0 |
| YarsRevenge      | [0, 32, 64, 96]                                 |              0 | [0, 1]                  |                    0 |
| Zaxxon           | [0, 8, 16, 24]                                  |              0 | [0]                     |                    0 |

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
