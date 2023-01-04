---
layout: "contents"
title: Gymnasium v0.26.3
---

# v0.26.3 Release Notes

Note: ale-py (atari) has not updated to Gymnasium yet. Therefore ``pip install gymnasium[atari]`` will fail, this will be fixed in ``v0.27``. In the meantime, use ``pip install shimmy[atari]`` for the fix.

## Bug Fixes
* Added Gym-Gymnasium compatibility converter to allow users to use Gym environments in Gymnasium by [@RedTachyon](https://github.com/RedTachyon) in https://github.com/Farama-Foundation/Gymnasium/pull/61
* Modify metadata in the ``HumanRendering`` and ``RenderCollection`` wrappers to have the correct metadata by [@RedTachyon](https://github.com/RedTachyon) in https://github.com/Farama-Foundation/Gymnasium/pull/35
* Simplified ``EpisodeStatisticsRecorder`` wrapper  by [@DavidSlayback](https://github.com/DavidSlayback) in https://github.com/Farama-Foundation/Gymnasium/pull/31
* Fix integer overflow in MultiDiscrete.flatten() by [@olipinski](https://github.com/olipinski) in https://github.com/Farama-Foundation/Gymnasium/pull/55
* Re-add the ability to specify the XML file for Mujoco environments by [@Kallinteris-Andreas](https://github.com/Kallinteris-Andreas) in https://github.com/Farama-Foundation/Gymnasium/pull/70

## Documentation change
* Add a tutorial for training an agent in Blackjack by [@till2](https://github.com/till2) in https://github.com/Farama-Foundation/Gymnasium/pull/64
* A very long list of documentation updates by [@mgoulao](https://github.com/mgoulao), [@vairodp](https://github.com/vairodp), [@WillDudley](https://github.com/WillDudley), [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) and [@jjshoots](https://github.com/jjshoots)

**Full Changelog**: https://github.com/Farama-Foundation/Gymnasium/compare/v0.26.2...v0.26.3

## Thank you for the new contributors
* [@vairodp](https://github.com/vairodp) made their first contribution in https://github.com/Farama-Foundation/Gymnasium/pull/41
* [@DavidSlayback](https://github.com/DavidSlayback) made their first contribution in https://github.com/Farama-Foundation/Gymnasium/pull/31
* [@WillDudley](https://github.com/WillDudley) made their first contribution in https://github.com/Farama-Foundation/Gymnasium/pull/51
* [@olipinski](https://github.com/olipinski) made their first contribution in https://github.com/Farama-Foundation/Gymnasium/pull/55
* [@jjshoots](https://github.com/jjshoots) made their first contribution in https://github.com/Farama-Foundation/Gymnasium/pull/58
* [@vmoens](https://github.com/vmoens) made their first contribution in https://github.com/Farama-Foundation/Gymnasium/pull/60
* [@till2](https://github.com/till2) made their first contribution in https://github.com/Farama-Foundation/Gymnasium/pull/64
* [@Kallinteris-Andreas](https://github.com/Kallinteris-Andreas) made their first contribution in https://github.com/Farama-Foundation/Gymnasium/pull/70

**Github Release**: https://github.com/Farama-Foundation/Gymnasium/releases/tag/v0.26.3
