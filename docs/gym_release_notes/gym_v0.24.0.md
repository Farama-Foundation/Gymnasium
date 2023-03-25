---
layout: "contents"
title: Gym v0.24.0
---

# v0.24.0 Release Notes

## Major changes

* Added v4 mujoco environments that use the new [deepmind mujoco](https://github.com/deepmind/mujoco) 2.2.0 module. This can be installed through ``pip install gym[mujoco]`` with the old bindings still being available using the ``v3`` environments and ``pip install gym[mujoco-py]``. These new ``v4`` environment should have the same training curves as ``v3``. For the Ant, we found that there was a contact parameter that was not applied in ``v3`` that can enabled in v4 however was found to produce significantly worse performance [see comment](https://github.com/openai/gym/pull/2762#issuecomment-1135362092) for more details. [@rodrigodelazcano](https://github.com/rodrigodelazcano)
* The vector environment step ``info`` API has been changes to allow hardware acceleration in the future. See [this PR](https://github.com/openai/gym/pull/2773) for the modified ``info`` style that now uses dictionaries instead of a list of environment info. If you still wish to use the list info style, then use the ``VectorListInfo`` wrapper. [@gianlucadecola](https://github.com/gianlucadecola)
* On ``gym.make``, the gym ``env_checker`` is run that includes calling the environment ``reset`` and ``step`` to check if the environment is compliant to the gym API. To disable this feature, run ``gym.make(..., disable_env_checker=True)``. [@RedTachyon](https://github.com/RedTachyon)
* Re-added ``gym.make("MODULE:ENV")`` import style that was accidentally removed in v0.22 [@arjun-kg](https://github.com/arjun-kg)
* ``Env.render`` is now order enforced such that ``Env.reset`` is required before ``Env.render`` is called. If this a required feature then set the ``OrderEnforcer`` wrapper ``disable_render_order_enforcing=True``. [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Added wind and turbulence to the Lunar Lander environment, this is by default turned off, use the ``wind_power`` and ``turbulence`` parameter. [@virgilt](https://github.com/virgilt)
* Improved the ``play`` function to allow multiple keyboard letter to pass instead of ascii value [@Markus28](https://github.com/Markus28)
* Added google style pydoc strings for most of the repositories [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) [@Markus28](https://github.com/Markus28)
* Added discrete car racing environment version through ``gym.make("CarRacing-v1", continuous=False)``
* Pygame is now an optional module for box2d and classic control environments that is only necessary for rendering. Therefore, install pygame using ``pip install gym[box2d]`` or ``pip install gym[classic_control]`` [@gianlucadecola](https://github.com/gianlucadecola) [@RedTachyon](https://github.com/RedTachyon)
* Fixed bug in batch spaces (used in ``VectorEnv``) such that the original space's seed was ignored [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Added ``AutoResetWrapper`` that automatically calls ``Env.reset`` when ``Env.step`` done is ``True`` [@balisujohn](https://github.com/balisujohn)

## Minor changes

* Bipedal Walker and Lunar Lander's observation spaces have non-infinite upper and lower bounds. [@jjshoots](https://github.com/jjshoots)
* Bumped the ALE-py version to ``0.7.5``
* Improved the performance of car racing through not rendering polygons off screen [@andrewtanJS](https://github.com/andrewtanJS)
* Fixed turn indicators that were black not red/white in Car racing [@jjshoots](https://github.com/jjshoots)
* Bug fixes for ``VecEnvWrapper`` to forward method calls to the environment [@arjun-kg](https://github.com/arjun-kg)
* Removed unnecessary try except on ``Box2d`` such that if ``Box2d`` is not installed correctly then a more helpful error is show [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Simplified the ``gym.registry`` backend [@RedTachyon](https://github.com/RedTachyon)
* Re-added python 3.6 support through backports of python 3.7+ modules. This is not tested or compatible with the mujoco environments. [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)

**Full Changelog**: https://github.com/openai/gym/compare/0.23.1...0.24.0

**Github Release**: https://github.com/openai/gym/releases/tag/0.24.0
