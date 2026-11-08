---
title: Atari
firstpage:
lastpage:
---

# Atari (moved to the Arcade Learning Environment)

```{warning}
The Atari environments and their documentation have moved to the
[Arcade Learning Environment (ALE)](https://ale.farama.org/). Gymnasium no longer
documents them here; every page below links to its new home on
[ale.farama.org](https://ale.farama.org/environments/).
```

The ALE is the Atari 2600 emulator these environments have always been built on. It
exposes them through the Gymnasium API itself, so they are created with `gymnasium.make`
exactly as before, under `ALE/` ids.

## Migration Guide

Install `ale-py`, either directly or through Gymnasium's `atari` extra:

```bash
pip install ale-py
# or
pip install "gymnasium[atari]"
```

Then register the environments with Gymnasium and make them as usual:

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

The common Atari observation pipeline stays in Gymnasium as
[`gymnasium.wrappers.AtariPreprocessing`](../api/wrappers/misc_wrappers).

## Environments

* [Adventure](https://ale.farama.org/environments/adventure/)
* [Air Raid](https://ale.farama.org/environments/air_raid/)
* [Alien](https://ale.farama.org/environments/alien/)
* [Amidar](https://ale.farama.org/environments/amidar/)
* [Assault](https://ale.farama.org/environments/assault/)
* [Asterix](https://ale.farama.org/environments/asterix/)
* [Asteroids](https://ale.farama.org/environments/asteroids/)
* [Atlantis](https://ale.farama.org/environments/atlantis/)
* [Atlantis2](https://ale.farama.org/environments/atlantis2/)
* [Backgammon](https://ale.farama.org/environments/backgammon/)
* [Bank Heist](https://ale.farama.org/environments/bank_heist/)
* [Basic Math](https://ale.farama.org/environments/basic_math/)
* [Battle Zone](https://ale.farama.org/environments/battle_zone/)
* [Beam Rider](https://ale.farama.org/environments/beam_rider/)
* [Berzerk](https://ale.farama.org/environments/berzerk/)
* [Blackjack](https://ale.farama.org/environments/blackjack/)
* [Bowling](https://ale.farama.org/environments/bowling/)
* [Boxing](https://ale.farama.org/environments/boxing/)
* [Breakout](https://ale.farama.org/environments/breakout/)
* [Carnival](https://ale.farama.org/environments/carnival/)
* [Casino](https://ale.farama.org/environments/casino/)
* [Centipede](https://ale.farama.org/environments/centipede/)
* [Chopper Command](https://ale.farama.org/environments/chopper_command/)
* [Crazy Climber](https://ale.farama.org/environments/crazy_climber/)
* [Crossbow](https://ale.farama.org/environments/crossbow/)
* [Darkchambers](https://ale.farama.org/environments/darkchambers/)
* [Defender](https://ale.farama.org/environments/defender/)
* [Demon Attack](https://ale.farama.org/environments/demon_attack/)
* [Donkey Kong](https://ale.farama.org/environments/donkey_kong/)
* [Double Dunk](https://ale.farama.org/environments/double_dunk/)
* [Earthworld](https://ale.farama.org/environments/earthworld/)
* [Elevator Action](https://ale.farama.org/environments/elevator_action/)
* [Enduro](https://ale.farama.org/environments/enduro/)
* [Entombed](https://ale.farama.org/environments/entombed/)
* [Et](https://ale.farama.org/environments/et/)
* [Fishing Derby](https://ale.farama.org/environments/fishing_derby/)
* [Flag Capture](https://ale.farama.org/environments/flag_capture/)
* [Freeway](https://ale.farama.org/environments/freeway/)
* [Frogger](https://ale.farama.org/environments/frogger/)
* [Frostbite](https://ale.farama.org/environments/frostbite/)
* [Galaxian](https://ale.farama.org/environments/galaxian/)
* [Gopher](https://ale.farama.org/environments/gopher/)
* [Gravitar](https://ale.farama.org/environments/gravitar/)
* [Hangman](https://ale.farama.org/environments/hangman/)
* [Haunted House](https://ale.farama.org/environments/haunted_house/)
* [Hero](https://ale.farama.org/environments/hero/)
* [Human Cannonball](https://ale.farama.org/environments/human_cannonball/)
* [Ice Hockey](https://ale.farama.org/environments/ice_hockey/)
* [Jamesbond](https://ale.farama.org/environments/jamesbond/)
* [Journey Escape](https://ale.farama.org/environments/journey_escape/)
* [Kaboom](https://ale.farama.org/environments/kaboom/)
* [Kangaroo](https://ale.farama.org/environments/kangaroo/)
* [Keystone Kapers](https://ale.farama.org/environments/keystone_kapers/)
* [King Kong](https://ale.farama.org/environments/king_kong/)
* [Klax](https://ale.farama.org/environments/klax/)
* [Koolaid](https://ale.farama.org/environments/koolaid/)
* [Krull](https://ale.farama.org/environments/krull/)
* [Kung Fu Master](https://ale.farama.org/environments/kung_fu_master/)
* [Laser Gates](https://ale.farama.org/environments/laser_gates/)
* [Lost Luggage](https://ale.farama.org/environments/lost_luggage/)
* [Mario Bros](https://ale.farama.org/environments/mario_bros/)
* [Miniature Golf](https://ale.farama.org/environments/miniature_golf/)
* [Montezuma Revenge](https://ale.farama.org/environments/montezuma_revenge/)
* [Mr Do](https://ale.farama.org/environments/mr_do/)
* [Ms Pacman](https://ale.farama.org/environments/ms_pacman/)
* [Name This Game](https://ale.farama.org/environments/name_this_game/)
* [Othello](https://ale.farama.org/environments/othello/)
* [Pacman](https://ale.farama.org/environments/pacman/)
* [Phoenix](https://ale.farama.org/environments/phoenix/)
* [Pitfall](https://ale.farama.org/environments/pitfall/)
* [Pitfall2](https://ale.farama.org/environments/pitfall2/)
* [Pong](https://ale.farama.org/environments/pong/)
* [Pooyan](https://ale.farama.org/environments/pooyan/)
* [Private Eye](https://ale.farama.org/environments/private_eye/)
* [Qbert](https://ale.farama.org/environments/qbert/)
* [Riverraid](https://ale.farama.org/environments/riverraid/)
* [Road Runner](https://ale.farama.org/environments/road_runner/)
* [Robotank](https://ale.farama.org/environments/robotank/)
* [Seaquest](https://ale.farama.org/environments/seaquest/)
* [Sir Lancelot](https://ale.farama.org/environments/sir_lancelot/)
* [Skiing](https://ale.farama.org/environments/skiing/)
* [Solaris](https://ale.farama.org/environments/solaris/)
* [Space Invaders](https://ale.farama.org/environments/space_invaders/)
* [Space War](https://ale.farama.org/environments/space_war/)
* [Star Gunner](https://ale.farama.org/environments/star_gunner/)
* [Superman](https://ale.farama.org/environments/superman/)
* [Surround](https://ale.farama.org/environments/surround/)
* [Tennis](https://ale.farama.org/environments/tennis/)
* [Tetris](https://ale.farama.org/environments/tetris/)
* [Tic Tac Toe 3D](https://ale.farama.org/environments/tic_tac_toe_3d/)
* [Time Pilot](https://ale.farama.org/environments/time_pilot/)
* [Trondead](https://ale.farama.org/environments/trondead/)
* [Turmoil](https://ale.farama.org/environments/turmoil/)
* [Tutankham](https://ale.farama.org/environments/tutankham/)
* [Up N Down](https://ale.farama.org/environments/up_n_down/)
* [Venture](https://ale.farama.org/environments/venture/)
* [Video Checkers](https://ale.farama.org/environments/video_checkers/)
* [Video Chess](https://ale.farama.org/environments/video_chess/)
* [Video Cube](https://ale.farama.org/environments/video_cube/)
* [Video Pinball](https://ale.farama.org/environments/video_pinball/)
* [Wizard Of Wor](https://ale.farama.org/environments/wizard_of_wor/)
* [Word Zapper](https://ale.farama.org/environments/word_zapper/)
* [Yars Revenge](https://ale.farama.org/environments/yars_revenge/)
* [Zaxxon](https://ale.farama.org/environments/zaxxon/)

## Links

* [Documentation](https://ale.farama.org/)
* [GitHub](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
* [PyPI](https://pypi.org/project/ale-py/)
