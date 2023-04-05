---
layout: "contents"
title: Gym v0.20.0
---

# v0.20.0 Release Notes

## Major Change

* Replaced Atari-Py dependency with ALE-Py and bumped all versions. This is a massive upgrade with many changes, please see the [full explainer](https://brosa.ca/blog/ale-release-v0.7) ([@JesseFarebro](https://github.com/JesseFarebro))
* Note that ALE-Py does not include ROMs. You can install ROMs in two lines of bash with ``AutoROM`` though (``pip3 install autorom and then autorom``), see https://github.com/PettingZoo-Team/AutoROM. This is the recommended approach for CI, etc.

## Breaking changes and new features:

* Add ``RecordVideo`` wrapper, deprecate ``monitor`` wrapper in favor of it and ``RecordEpisodeStatistics`` wrapper ([@vwxyzjn](https://github.com/vwxyzjn))
* Dependencies used outside of environments (e.g. for wrappers) are now in ``gym[other]`` ([@jkterry1](https://github.com/jkterry1))
* Moved algorithmic and unused toy-text envs (guessing game, hotter colder, nchain, roulette, kellycoinflip) to third party repos ([@jkterry1](https://github.com/jkterry1), [@Rohan138](https://github.com/Rohan138))
* Fixed flatten utility and flatdim in MultiDiscrete space ([@tristandeleu](https://github.com/tristandeleu))
* Add ``__setitem__`` to dict space ([@jfpettit](https://github.com/jfpettit))
* Large fixes to ``.contains`` method for box space ([@FirefoxMetzger](https://github.com/FirefoxMetzger))
* Made blackjack environment properly comply with Barto and Sutton book standard, bumped to v1 ([@RedTachyon](https://github.com/RedTachyon))
* Added ``NormalizeObservation`` and ``NormalizeReward`` wrappers ([@vwxyzjn](https://github.com/vwxyzjn))
* Add ``__getitem__`` and ``__len__`` to MultiDiscrete space ([@XuehaiPan](https://github.com/XuehaiPan))
* Changed ``.shape`` to be a property of box space to prevent unexpected behaviors ([@RedTachyon](https://github.com/RedTachyon))

## Bug fixes and upgrades

* Video recorder gracefully handles closing ([@XuehaiPan](https://github.com/XuehaiPan))
* Remaining unnecessary dependencies in setup.py are resolved ([@jkterry1](https://github.com/jkterry1))
* Minor acrobot performance improvements ([@TuckerBMorgan](https://github.com/TuckerBMorgan))
* Pendulum properly renders when 0 force is sent ([@Olimoyo](https://github.com/Olimoyo))
* Make observations dtypes be consistent with observation space dtypes for all classic control envs and bipedal-walker ([@RedTachyon](https://github.com/RedTachyon))
* Removed unused and long deprecated features in registration ([@Rohan138](https://github.com/Rohan138))
* Framestack wrapper now inherits from obswrapper ([@jfpettit](https://github.com/jfpettit))
* Seed method for ``spaces.Tuple`` and ``spaces.Dict`` now properly function, are fully stochastic, are fully featured and behave in the expected manner ([@XuehaiPan](https://github.com/XuehaiPan), [@RaghuSpaceRajan](https://github.com/RaghuSpaceRajan))
* Replace ``time()`` with ``perf_counter()`` for better measurements of short duration ([@zuoxingdong](https://github.com/zuoxingdong))

**Full Changelog**: https://github.com/openai/gym/compare/0.19.0...v0.20.0

**Github Release**: https://github.com/openai/gym/releases/tag/v0.20.0
