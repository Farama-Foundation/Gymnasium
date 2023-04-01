---
layout: "contents"
title: Gym v0.25.1
---

# 0.25.1 Release Notes

* Added rendering for CliffWalking environment [@younik](https://github.com/younik)
* ``PixelObservationWrapper`` only supports the new render API due to difficulty in supporting both old and new APIs. A warning is raised if the user is using the old API [@vmoens](https://github.com/vmoens)

## Bug fix

* Revert an incorrect edition on wrapper.FrameStack [@ZhiqingXiao](https://github.com/ZhiqingXiao)
* Fix reset bounds for mountain car [@psc-g](https://github.com/psc-g)
* Removed skipped tests causing bugs not to be caught [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Added backward compatibility for environments without metadata [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Fixed ``BipedalWalker`` rendering for RGB arrays [@1b15](https://github.com/1b15)
* Fixed bug in ``PixelObservationWrapper`` for using the new rendering [@younik](https://github.com/younik)

## Typos

* Rephrase observations' definition in Lunar Lander Environment [@EvanMath](https://github.com/EvanMath)
* Top-docstring in ``gym/spaces/dict.py`` [@Ice1187](https://github.com/Ice1187)
* Several typos in ``humanoidstandup_v4.py``, ``mujoco_env.py``, and ``vector_list_info.py`` [@timgates42](https://github.com/timgates42)
* Typos in passive environment checker [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Typos in Swimmer rotations [@lin826](https://github.com/lin826)

**Full Changelog**: https://github.com/openai/gym/compare/0.25.0...0.25.1

**Github Release**: https://github.com/openai/gym/releases/tag/0.25.1
