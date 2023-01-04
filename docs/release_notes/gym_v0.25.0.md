---
layout: "contents"
title: Gym v0.25.0
---

# v0.25.0 Release notes

This release finally introduces all new API changes that have been planned for the past year or more, all of which will be turned on by default in a subsequent release. After this point, Gym development should get massively smoother. This release also fixes large bugs present in [0.24.0](https://github.com/openai/gym/releases/tag/0.24.0) and [0.24.1](https://github.com/openai/gym/releases/tag/0.24.1), and we highly discourage using those releases.

## API Changes

* ``Env.step`` - A majority of deep reinforcement learning algorithm implementations are incorrect due to an important difference in theory and practice as ``done`` is not equivalent to ``termination``. As a result, we have modified the ``step`` function to return five values: ``obs, reward, termination, truncation, info``. The full theoretical and practical reason (along with example code changes) for these changes will be explained in a soon-to-be-released blog post. The aim for the change to be backward compatible (for now), for issues, please put report the issue on github or the discord. [@arjun-kg](https://github.com/arjun-kg)
* ``Env.Render`` - The render API is changed such that the mode has to be specified during ``gym.make`` with the keyword ``render_mode``, after which, the render mode is fixed. For further details see https://younis.dev/blog/2022/render-api/ and [#2671](https://github.com/openai/gym/pull/2671). This has the additional changes
  * with ``render_mode="human"`` you don't need to call ``.render()``, rendering will happen automatically on ``env.step()``
  * with ``render_mode="rgb_array"``, ``.render()`` pops the list of frames rendered since the last ``.reset()``
  * with ``render_mode="single_rgb_array"``, ``.render()`` returns a single frame, like before.
* ``Space.sample(mask=...)`` allows a mask when sampling actions to enable/disable certain actions from being randomly sampled. We recommend developers add this to the ``info`` parameter returned by ``Env.reset(return_info=True)`` and ``Env.step``. See [#2906](https://github.com/openai/gym/pull/2906) for example implementations of the masks or the individual spaces. We have added an example version of this in the taxi environment. [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Add ``Graph`` space for environments that use graph style observation or action spaces. Currently, the ``node`` and ``edge`` spaces can only be ``Box`` or ``Discrete`` spaces. [@jjshoots](https://github.com/jjshoots)
* Add ``Text`` space for Reinforcement Learning that involves communication between agents and have dynamic length messages (otherwise ``MultiDiscrete`` can be used). [@ryanrudes](https://github.com/ryanrudes) [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)

## Bug fixes

* Fixed car racing termination where if the agent finishes the final lap, then the environment ends through truncation not termination. This added a version bump to Car racing to v2 and removed Car racing discrete in favour of ``gym.make("CarRacing-v2", continuous=False)`` [@araffin](https://github.com/araffin)
* In ``v0.24.0``, ``opencv-python`` was an accidental requirement for the project. This has been reverted. [@KexianShen](https://github.com/KexianShen) [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Updated ``utils.play`` such that if the environment specifies ``keys_to_action``, the function will automatically use that data. [@Markus28](https://github.com/Markus28)
* When rendering the blackjack environment, fixed bug where rendering would change the dealers top car. [@balisujohn](https://github.com/balisujohn)
* Updated mujoco docstring to reflect changes that were accidentally overwritten. [@Markus28](https://github.com/Markus28)

## Misc

* The whole project is partially type hinted using [pyright](https://github.com/microsoft/pyright) (none of the project files is ignored by the type hinter). [@RedTachyon](https://github.com/RedTachyon) [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) (Future work will add strict type hinting to the core API)
* Action masking added to the taxi environment (no version bump due to being backwards compatible) [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* The ``Box`` space shape inference is allows ``high`` and ``low`` scalars to be automatically set to ``(1,)`` shape. Minor changes to identifying scalars. [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Added option support in classic control environment to modify the bounds on the initial random state of the environment [@psc-g](https://github.com/psc-g)
* The ``RecordVideo`` wrapper is becoming deprecated with no support for ``TextEncoder`` with the new render API. The plan is to replace ``RecordVideo`` with a single function that will receive a list of frames from an environment and automatically render them as a video using [MoviePy](https://github.com/Zulko/moviepy). [@johnMinelli](https://github.com/johnMinelli)
* The gym ``py.Dockerfile`` is optimised from 2Gb to 1.5Gb through a number of optimisations [@TheDen](https://github.com/TheDen)

**Full Changelog**: https://github.com/openai/gym/compare/0.24.1...0.25.0

**Github Release**: https://github.com/openai/gym/releases/tag/0.25.0
