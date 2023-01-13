---
layout: "contents"
title: Gym v0.24.1
---

# v0.24.1 Release Notes

This is a bug fix release for version 0.24.0

## Bugs fixed

* Replaced the environment checker introduced in V24, such that the environment checker will not call ``Env.step`` and ``Env.reset`` during ``gym.make``. This new version is a wrapper that will observe the data that ``Env.step`` and ``Env.reset`` returns on their first call and check the data against the environment checker. [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Fixed MuJoCo v4 arguments key callback, closing the environment in renderer and the ``mujoco_rendering.close`` method. [@rodrigodelazcano](https://github.com/rodrigodelazcano)
* Removed redundant warning in registration [@RedTachyon](https://github.com/RedTachyon)
* Removed maths operations from MuJoCo xml files [@quagla](https://github.com/quagla)
* Added support for unpickling legacy ``spaces.Box`` [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Fixed mujoco environment action and observation space docstring tables [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Disable wrappers from accessing ``_np_random`` property and ``np_random`` is now forwarded to environments [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Rewrite setup.py to add a ``"testing"`` meta dependency group [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Fixed docstring in ``rescale_action`` wrapper [@gianlucadecola](https://github.com/gianlucadecola)

**Full Changelog**: https://github.com/openai/gym/compare/0.24.0...0.24.1

**Github Release**: https://github.com/openai/gym/releases/tag/0.24.1
