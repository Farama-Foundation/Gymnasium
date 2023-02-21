---
layout: "contents"
title: Gym v0.19.0
---

# v0.19.0 Release Notes

Gym 0.19.0 is a large maintenance release, and the first since [@jkterry1](https://github.com/jkterry1) became the maintainer. There should be no breaking changes in this release.

## New features

* Added custom datatype argument to multidiscrete space ([@m-orsini](https://github.com/m-orsini))
* API compliance test added based on SB3 and PettingZoo tests ([@amtamasi](https://github.com/amtamasi))
* RecordEpisodeStatics works with VectorEnv ([@vwxyzjn](https://github.com/vwxyzjn))

## Bug fixes

* Removed unused dependencies, removed unnescesary dependency version requirements that caused installation issues on newer machines, added full requirements.txt and moved general dependencies to extras. Notably, "toy_text" is not a used extra. atari-py is now pegged to a precise working version pending the switch to ale-py ([@jkterry1](https://github.com/jkterry1))
* Bug fixes to rewards in FrozenLake and FrozenLake8x8; versions bumped to v1 ([@ZhiqingXiao](https://github.com/ZhiqingXiao))
* Removed remaining numpy depreciation warnings ([@super-pirata](https://github.com/super-pirata))
* Fixes to video recording ([@mahiuchun](https://github.com/mahiuchun), [@zlig](https://github.com/zlig))
* EZ pickle argument fixes ([@zzyunzhi](https://github.com/zzyunzhi), [@jamesborg46](https://github.com/jamesborg46))
* Other very minor (nonbreaking) fixes

## Other

* Removed small bits of dead code ([@jkterry1](https://github.com/jkterry1))
* Numerous typo, CI and documentation fixes (mostly [@cclauss](https://github.com/cclauss))
* New readme and updated third party env list ([@jkterry1](https://github.com/jkterry1))
* Code is now all flake8 compliant through black ([@cclauss](https://github.com/cclauss))

**Full Changelog**: https://github.com/openai/gym/compare/0.18.3...0.19.0

**Github Release**: https://github.com/openai/gym/releases/tag/0.19.0
