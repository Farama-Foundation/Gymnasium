---
layout: "contents"
title: Gym v0.26.1
---

# v0.26.1 Release Notes

This is a very minor bug fix release for 0.26.0

## Bug Fixes
* [#3072](https://github.com/openai/gym/pull/3072) - Previously mujoco was a necessary module even if only ``mujoco-py`` was used. This has been fixed to allow only ``mujoco-py`` to be installed and used. [@YouJiacheng](https://github.com/YouJiacheng)
* [#3076](https://github.com/openai/gym/pull/3076) - ``PixelObservationWrapper`` raises an exception if the ``env.render_mode`` is not specified. [@vmoens](https://github.com/vmoens)
* [#3080](https://github.com/openai/gym/pull/3080) - Fixed bug in ``CarRacing`` where the colour of the wheels were not correct [@foxik](https://github.com/foxik)
* [#3083](https://github.com/openai/gym/pull/3083) - Fixed ``BipedalWalker`` where if the agent moved backwards then the rendered arrays would be a different size. [@younik](https://github.com/younik)

## Spelling
* Fixed truncation typo in readme API example [@rdnfn](https://github.com/rdnfn)
* Updated pendulum observation space from angle to theta to make more consistent [@ikamensh](https://github.com/ikamensh)

**Full Changelog**: https://github.com/openai/gym/compare/0.26.0...0.26.1

**Github Release**: https://github.com/openai/gym/releases/tag/0.26.1
