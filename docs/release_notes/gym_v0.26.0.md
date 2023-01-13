---
layout: "contents"
title: Gym v0.26.0
---

# v0.26.0 Release Notes

This release is aimed to be the last of the major API changes to the core API. All previously "turned off" changes of the base API (step termination / truncation, reset info, no seed function, render mode determined by initialization) are now expected by default. We still plan to make breaking changes to Gym itself, but to things that are very easy to upgrade (environments and wrappers), and things that aren't super commonly used (the vector API). Once those aspects are stabilized, we'll do a proper 1.0 release and follow semantic versioning. Additionally, unless something goes terribly wrong with this release, and we have to release a patched version, this will be the last release of Gym for a while.

If you've been waiting for a "stable" release of Gym to upgrade your project given all the changes that have been going on, this is the one.

We also just wanted to say that we tremendously appreciate the communities patience with us as we've gone on this journey taking over the maintenance of Gym and making all of these huge changes to the core API. We appreciate your patience and support, but hopefully, all the changes from here on out will be much more minor.

## Breaking backward compatibility

These changes are true of all gym's internal wrappers and environments but for environments not updated, we provide the ``EnvCompatibility``  wrapper for users to convert old gym v21 / 22 environments to the new core API. This wrapper can be easily applied in ``gym.make`` and ``gym.register`` through the ``apply_api_compatibility``  parameters.

* ``Step`` Termination / truncation - The ``Env.step`` function returns 5 values instead of 4 previously (observations, reward, termination, truncation, info). A blog with more details will be released soon to explain this decision. [@arjun-kg](https://github.com/arjun-kg)
* Reset info - The ``Env.reset`` function returns two values (``obs`` and ``info``) with no ``return_info`` parameter for gym wrappers and environments. This is important for some environments that provided action masking information for each actions which was not possible for resets. [@balisujohn](https://github.com/balisujohn)
* No Seed function - While ``Env.seed`` was a helpful function, this was almost solely used for the beginning of the episode and is added to ``gym.reset(seed=...)``. In addition, for several environments like Atari that utilise external random number generators, it was not possible to set the seed at any time other than ``reset``. Therefore, ``seed`` is no longer expected to function within gym environments and is removed from all gym environments [@balisujohn](https://github.com/balisujohn)
* Rendering - It is normal to only use a single render mode and to help open and close the rendering window, we have changed ``Env.render`` to not take any arguments and so all render arguments can be part of the environment's constructor i.e., ``gym.make("CartPole-v1", render_mode="human")``. For more detail on the new API, see [blog post](https://younis.dev/blog/render-api/) [@younik](https://github.com/younik)

## Major changes
* Render modes - In ``v0.25``, there was a change in the meaning of render modes, i.e. ``"rgb_array"`` returned a list of rendered frames with ``"single_rgb_array"`` returned a single frame. This has been reverted in this release with ``"rgb_array"`` having the same meaning as previously to return a single frame with a new mode ``"rgb_array_list"`` returning a list of RGB arrays. The capability to return a list of rendering observations achieved through a wrapper applied during ``gym.make``. [#3040](https://github.com/openai/gym/pull/3040) [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) [@younik](https://github.com/younik)
* Added ``save_video`` function that uses ``MoviePy`` to render a list of RGB frames and updated ``RecordVideo`` to use this function. This removes support for recording ``ansi`` outputs. [#3016](https://github.com/openai/gym/pull/3016) [@younik](https://github.com/younik)
* Random Number Generator functions (``seeding.np_random``): ``rand``, ``randn``, ``randint``, ``get_state``, ``set_state``, ``hash_seed``, ``create_seed``, ``_bigint_from_bytes`` and ``_int_list_from_bigint`` have been removed. [@balisujohn](https://github.com/balisujohn)
* Bump ``ale-py`` to ``0.8.0`` which is compatibility with the new core API
* Added ``EnvAPICompatibility`` wrapper [@RedTachyon](https://github.com/RedTachyon)

## Minor changes

* Added improved ``Sequence``, ``Graph`` and ``Text`` sample masking [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Improved the ``gym.make`` and ``gym.register`` type hinting with ``entry_point`` being a necessary parameter of ``gym.register``. [#3041](https://github.com/openai/gym/pull/3041) [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Changed all URL to the new gym website https://www.gymlibrary.dev/ [@FieteO](https://github.com/FieteO)
* Fixed mujoco offscreen rendering with weight and height value > 500 [#3044](https://github.com/openai/gym/pull/3044) [@YouJiacheng](https://github.com/YouJiacheng)
* Allowed toy_text environment to render on headless machines [#3037](https://github.com/openai/gym/pull/3037) [@RedTachyon](https://github.com/RedTachyon)
* Renamed the motors in the mujoco swimmer envs [#3036](https://github.com/openai/gym/pull/3036) [@lin826](https://github.com/lin826)

**Full Changelog**: https://github.com/openai/gym/compare/0.25.2...0.26.0

**Github Release**: https://github.com/openai/gym/releases/tag/0.26.0
