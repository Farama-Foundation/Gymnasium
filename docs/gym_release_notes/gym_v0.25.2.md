---
layout: "contents"
title: Gym v0.25.2
---

# v0.25.2 Release Notes

This is a fairly minor bug fix release.

## Bug Fixes

* Removes requirements for ``_TimeLimit.truncated`` in info for step compatibility functions. This makes the step compatible with Envpool [@arjun-kg](https://github.com/arjun-kg)
* As the ordering of ``Dict`` spaces matters when flattening spaces, updated the ``__eq__`` to account for the ``.keys()`` ordering. [@XuehaiPan](https://github.com/XuehaiPan)
* Allows ``CarRacing`` environment to be pickled. Updated all gym environments to be correctly pickled. [@RedTachyon](https://github.com/RedTachyon)
* Seeding ``spaces.Dict`` and ``spaces.Tuple`` spaces with integers can cause lower-specification computers to hang due to requiring 8Gb memory. Updated the seeding with integers to not require unique subseeds (subseed collisions are rare). For users that require unique subseeds for all subspaces, we recommend using a dictionary or tuple with the subseeds. [@olipinski](https://github.com/olipinski)
* Fixed the metaclass implementation for the new render api to allow custom environments to use metaclasses as well. [@YouJiacheng](https://github.com/YouJiacheng)

## Updates

* Simplifies the step compatibility functions to make them easier to debug. ``TimeLimit`` wrapper with the old step API favours terminated over truncated if both are true. This is as the old done step API can only encode 3 states (cannot encode ``terminated=True`` and ``truncated=True``) therefore we must encode to only ``terminated=True`` or ``truncated=True``. [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Add Swig as a dependency of Box2d [@kir0ul](https://github.com/kir0ul)
* Add type annotation for ``render_mode`` and ``metadata`` [@bkrl](https://github.com/bkrl)

**Full Changelog**: https://github.com/openai/gym/compare/0.25.1...0.25.2

**Github Release**: https://github.com/openai/gym/releases/tag/0.25.2
