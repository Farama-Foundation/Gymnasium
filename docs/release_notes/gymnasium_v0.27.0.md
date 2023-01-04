---
layout: "contents"
title: Gymnasium v0.27.0
---

# v0.27.0 Release Notes

Gymnasium 0.27.0 is our first major release of Gymnasium. It has several significant new features, and numerous small bug fixes and code quality improvements as we work through our backlog. There should be no breaking changes beyond dropping Python 3.6 support and remove the mujoco ``Viewer`` class in favor of a ``MujocoRendering`` class. You should be able to upgrade your code that's using Gymnasium 0.26.x to 0.27.0 with little-to-no-effort.

Like always, our development roadmap is publicly available [here](https://github.com/Farama-Foundation/Gymnasium/issues/12) so you can follow our future plans. The only large breaking changes that are still planned are switching selected environments to use hardware accelerated physics engines and our long standing plans for overhauling the vector API and built-in wrappers.

This release notably includes an entirely new part of the library: ``gymnasium.experimental``. We are adding new features, wrappers and functional environment API discussed below for users to test and try out to find bugs and provide feedback.

## New Wrappers

These new wrappers, accessible in ``gymnasium.experimental.wrappers``, see the full list in https://gymnasium.farama.org/main/api/experimental/ are aimed to replace the wrappers in gymnasium v0.30.0 and contain several improvements
* (Work in progress) Support arbitrarily complex observation / action spaces. As RL has advanced, action and observation spaces are becoming more complex and the current wrappers were not implemented with this mind.
* Support for Jax-based environments. With hardware accelerated environments, i.e. Brax, written in Jax and similar PyTorch based programs, NumPy is not the only game in town anymore for writing environments. Therefore, these upgrades will use [Jumpy](https://github.com/farama-Foundation/jumpy), a project developed by Farama Foundation to provide automatic compatibility for NumPy, Jax and in the future PyTorch data for a large subset of the NumPy functions.
* More wrappers. Projects like [Supersuit](https://github.com/farama-Foundation/supersuit) aimed to bring more wrappers for RL, however, many users were not aware of the wrappers, so we plan to move the wrappers into Gymnasium. If we are missing common wrappers from the list provided above, please create an issue and we would be interested in adding it.
* Versioning. Like environments, the implementation details of wrappers can cause changes in agent performance. Therefore, we propose adding version numbers to all wrappers, i.e., ``LambaActionV0``. We don't expect these version numbers to change regularly and will act similarly to environment version numbers. This should ensure that all users know when significant changes could affect your agent's performance for environments and wrappers. Additionally, we hope that this will improve reproducibility of RL in the future, which is critical for academia.
* In v28, we aim to rewrite the VectorEnv to not inherit from Env, as a result new vectorized versions of the wrappers will be provided.

Core developers: [@gianlucadecola](https://github.com/gianlucadecola), [@RedTachyon](https://github.com/RedTachyon), [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)

## Functional API

The ``Env`` class provides a very generic structure for environments to be written in allowing high flexibility in the program structure. However, this limits the ability to efficiently vectorize environments, compartmentalize the environment code, etc. Therefore, the ``gymnasium.experimental.FuncEnv`` provides a much more strict structure for environment implementation with stateless functions, for every stage of the environment implementation. This class does not inherit from ``Env`` and requires a translation / compatibility class for doing this. We already provide a ``FuncJaxEnv`` for converting jax-based ``FuncEnv`` to ``Env``. We hope this will help improve the readability of environment implementations along with potential speed-ups for users that vectorize their code.

This API is very experimental so open to changes in the future. We are interested in feedback from users who try to use the API which we believe will be in particular interest to users exploring RL planning, model-based RL and modifying environment functions like the rewards.

Core developers: [@RedTachyon](https://github.com/RedTachyon), [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts), [@balisujohn](https://github.com/balisujohn)

## Other Major changes

* Refactor Mujoco Rendering mechanisms to use a separate thread for OpenGL. Remove ``Viewer`` in favor of ``MujocoRenderer`` which offscreen, human and other render mode can use by [@rodrigodelazcano](https://github.com/rodrigodelazcano) in [#112](https://github.com/Farama-Foundation/Gymnasium/pull/112)
* Add deprecation warning to ``gym.make(..., apply_env_compatibility=True)`` in favour of ``gym.make("GymV22Environment", env_id="...")`` by [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) in [#125](https://github.com/Farama-Foundation/Gymnasium/pull/125)
* Add ``gymnasium.pprint_registry()`` for pretty printing the gymnasium registry by [@kad99kev](https://github.com/kad99kev) in [#124](https://github.com/Farama-Foundation/Gymnasium/pull/124)
* Changes discrete dtype to np.int64 such that samples are np.int64 not python ints. by [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) in [#141](https://github.com/Farama-Foundation/Gymnasium/pull/141)
* Add migration guide for OpenAI Gym v21 to v26 by [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) in https://github.com/Farama-Foundation/Gymnasium/pull/72
* Add complete type hinting of ``core.py`` for ``Env``, ``Wrapper`` and more by [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) in [#39](https://github.com/Farama-Foundation/Gymnasium/pull/39)
* Add complete type hinting for all spaces in ``gymnasium.spaces`` by [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) in [#37](https://github.com/Farama-Foundation/Gymnasium/pull/37)
* Make window in ``play()`` resizable by [@Markus28](https://github.com/Markus28) in [#198](https://github.com/Farama-Foundation/Gymnasium/pull/190)
* Add REINFORCE implementation tutorial by [@siddarth-c](https://github.com/siddarth-c) in [#155](https://github.com/Farama-Foundation/Gymnasium/pull/155)

## Bug fixes and documentation changes

* Remove auto close in ``VideoRecorder`` wrapper by [@younik](https://github.com/younik) in [#42](https://github.com/Farama-Foundation/Gymnasium/pull/42)
* Change ``seeding.np_random`` error message to report seed type by [@theo-brown](https://github.com/theo-brown) in [#74](https://github.com/Farama-Foundation/Gymnasium/pull/74)
* Include shape in MujocoEnv error message by [@ikamensh](https://github.com/ikamensh) in [#83](https://github.com/Farama-Foundation/Gymnasium/pull/83)
* Add pretty Feature/GitHub issue form by [@tobirohrer](https://github.com/tobirohrer) in [#89](https://github.com/Farama-Foundation/Gymnasium/pull/89)
* Added testing for the render return data in ``check_env`` and ``PassiveEnvChecker`` by [@Markus28](https://github.com/Markus28) in [#117](https://github.com/Farama-Foundation/Gymnasium/pull/117)
* Fix docstring and update action space description for classic control environments by [@Thytu](https://github.com/Thytu) in [#123](https://github.com/Farama-Foundation/Gymnasium/pull/123)
* Fix ``__all__`` in root ``__init__.py`` to specify the correct folders by [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) in [#130](https://github.com/Farama-Foundation/Gymnasium/pull/130)
* Fix ``play()`` assertion error by [@Markus28](https://github.com/Markus28) in [#132](https://github.com/Farama-Foundation/Gymnasium/pull/132)
* Update documentation for Frozen Lake ``is_slippy`` by [@MarionJS](https://github.com/MarionJS) in [#136](https://github.com/Farama-Foundation/Gymnasium/pull/136)
* Fixed warnings when ``render_mode`` is None by [@younik](https://github.com/younik) in [#143](https://github.com/Farama-Foundation/Gymnasium/pull/143)
* Added ``is_np_flattenable`` property to documentation by [@Markus28](https://github.com/Markus28) in [#172](https://github.com/Farama-Foundation/Gymnasium/pull/172)
* Updated Wrapper documentation by [@Markus28](https://github.com/Markus28) in [#173](https://github.com/Farama-Foundation/Gymnasium/pull/173)
* Updated formatting of spaces documentation by [@Markus28](https://github.com/Markus28) in [#174](https://github.com/Farama-Foundation/Gymnasium/pull/174)
* For FrozenLake, add seeding in random map generation by [@kir0ul](https://github.com/kir0ul) in [#139](https://github.com/Farama-Foundation/Gymnasium/pull/139)
* Add notes for issues when unflattening samples from flattened spaces by [@rusu24edward](https://github.com/rusu24edward) in [#164](https://github.com/Farama-Foundation/Gymnasium/pull/164)
* Include pusher environment page to website by [@axb2035](https://github.com/axb2035) in [#171](https://github.com/Farama-Foundation/Gymnasium/pull/171)
* Add check in ``AsyncVectorEnv`` for success before splitting result in ``step_wait`` by [@aaronwalsman](https://github.com/aaronwalsman) in [#178](https://github.com/Farama-Foundation/Gymnasium/pull/178)
* Add documentation for ``MuJoCo.Ant-v4.use_contact_forces`` by [@Kallinteris-Andreas](https://github.com/Kallinteris-Andreas) in [#183](https://github.com/Farama-Foundation/Gymnasium/pull/183)
* Fix typos in README.md by [@cool-RR](https://github.com/cool-RR) in [#184](https://github.com/Farama-Foundation/Gymnasium/pull/184)
* Add documentation for ``MuJoCo.Ant`` v4 changelog by [@Kallinteris-Andreas](https://github.com/Kallinteris-Andreas) in [#186](https://github.com/Farama-Foundation/Gymnasium/pull/186)
* Fix ``MuJoCo.Ant`` action order in documentation by [@Kallinteris-Andreas](https://github.com/Kallinteris-Andreas) in [#208](https://github.com/Farama-Foundation/Gymnasium/pull/208)
* Add ``raise-from`` exception for the whole codebase by [@cool-RR](https://github.com/cool-RR) in [#205](https://github.com/Farama-Foundation/Gymnasium/pull/205)

## Behind-the-scenes changes
* Docs Versioning by [@mgoulao](https://github.com/mgoulao) in [#73](https://github.com/Farama-Foundation/Gymnasium/pull/73)
* Added Atari environments to tests, removed dead code by [@Markus28](https://github.com/Markus28) in [#78](https://github.com/Farama-Foundation/Gymnasium/pull/78)
* Fix missing build steps in versioning workflows by [@mgoulao](https://github.com/mgoulao) in [#81](https://github.com/Farama-Foundation/Gymnasium/pull/81)
* Small improvements to environments pages by [@mgoulao](https://github.com/mgoulao) in [#110](https://github.com/Farama-Foundation/Gymnasium/pull/110)
* Update the third-party environment documentation by [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) in [#138](https://github.com/Farama-Foundation/Gymnasium/pull/138)
* Update docstrings for improved documentation by [@axb2035](https://github.com/axb2035) in [#160](https://github.com/Farama-Foundation/Gymnasium/pull/160)
* Test core dependencies in CI by [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts) in [#146](https://github.com/Farama-Foundation/Gymnasium/pull/146)
* Update and rerun ``pre-commit`` hooks for better code quality by [@XuehaiPan](https://github.com/XuehaiPan) in [#179](https://github.com/Farama-Foundation/Gymnasium/pull/179)

**Full Changelog**: https://github.com/Farama-Foundation/Gymnasium/compare/v0.26.3...v0.27.0

**Github Release**: https://github.com/Farama-Foundation/Gymnasium/releases/tag/v0.27.0
