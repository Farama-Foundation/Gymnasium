---
layout: "contents"
title: Gym v0.22.0
---

# v0.22 Release Notes

This release represents the largest set of changes ever to Gym, and represents a huge step towards the plans for 1.0 outlined here: [#2524](https://github.com/openai/gym/pull/2524)

Gym now has a new comprehensive documentation site: https://www.gymlibrary.ml/ !

## API changes

* ``Env.reset`` now accepts three new arguments:
 * ``options``: Usable for things like controlling curriculum learning without reinitializing the environment, which can be expensive ([@RedTachyon](https://github.com/RedTachyon))
 * ``seed``: Environment seeds can be passed to this reset argument in the future. The old ``.seed()`` method is being deprecated in favor of this, though it will continue to function as before until the 1.0 release for backwards compatibility purposes ([@RedTachyon](https://github.com/RedTachyon))
 * ``return_info``: when set to ``True``, reset will return obs, info. This currently defaults to ``False``, but will become the default behavior in Gym 1.0 ([@RedTachyon](https://github.com/RedTachyon))

* Environment names no longer require a version during registration and will suggest intelligent similar names ([@kir0ul](https://github.com/kir0ul), [@JesseFarebro](https://github.com/JesseFarebro))

* Vector environments now support terminal_observation in ``info`` and support batch action spaces ([@vwxyzjn](https://github.com/vwxyzjn), [@tristandeleu](https://github.com/tristandeleu))

## Environment changes

* The blackjack and frozen lake toy_text environments now have nice graphical rendering using PyGame ([@1b15](https://github.com/1b15))
* Moved robotics environments to gym-robotics package ([@seungjaeryanlee](https://github.com/seungjaeryanlee), [@Rohan138](https://github.com/Rohan138), [@vwxyzjn](https://github.com/vwxyzjn)) (per discussion in [#2456](https://github.com/openai/gym/pull/2456) (comment))
* The bipedal walker and lunar lander environments were consolidated into one class ([@andrewtanJS](https://github.com/andrewtanJS))
* Atari environments now use standard seeding API ([@JesseFarebro](https://github.com/JesseFarebro))
* Fixed large bug fixes in car_racing box2d environment, bumped version ([@carlosluis](https://github.com/carlosluis), [@araffin](https://github.com/araffin))
* Refactored all box2d and classic_control environments to use PyGame instead of Pyglet as issues with pyglet has been one of the most frequent sources of GitHub issues over the life of the gym project ([@andrewtanJS](https://github.com/andrewtanJS))

## Other changes

* Removed DiscreteEnv class, built in environments no longer use it ([@carlosluis](https://github.com/carlosluis))
* Large numbers of type hints added ([@ikamensh](https://github.com/ikamensh), [@RedTachyon](https://github.com/RedTachyon))
* Python 3.10 support
* Tons of additional code refactoring, cleanup, error message improvements and small bug fixes ([@vwxyzjn](https://github.com/vwxyzjn), [@Markus28](https://github.com/Markus28), [@RushivArora](https://github.com/RushivArora), [@jjshoots](https://github.com/jjshoots), [@XuehaiPan](https://github.com/XuehaiPan), [@Rohan138](https://github.com/Rohan138), [@JesseFarebro](https://github.com/JesseFarebro), [@Ericonaldo](https://github.com/Ericonaldo), [@AdilZouitine](https://github.com/AdilZouitine), [@RedTachyon](https://github.com/RedTachyon))
* All environment files now have dramatically improved readmes at the top (that the documentation website automatically pulls from)
* As part of the seeding changes, Gym's RNG has been modified to use the ``np.random.Generator`` as the RandomState API has been deprecated. The methods ``randint``, ``rand``, ``randn`` are replaced by ``integers``, ``random`` and ``standard_normal`` respectively. As a consequence, the random number generator has changed from ``MT19937`` to ``PCG64``.

**Full Changelog**: https://github.com/openai/gym/compare/v0.21.0...0.22.0

**Github Release**: https://github.com/openai/gym/releases/tag/0.22.0
