---
layout: "contents"
title: Gym v0.21.0
---

# v0.21.0 Release Notes

* The old Atari entry point that was broken with the last release and the upgrade to ALE-Py is fixed ([@JesseFarebro](https://github.com/JesseFarebro))
* Atari environments now give much clearer error messages and warnings ([@JesseFarebro](https://github.com/JesseFarebro))
* A new plugin system to enable an easier inclusion of third party environments has been added ([@JesseFarebro](https://github.com/JesseFarebro))
* Atari environments now use the new plugin system to prevent clobbered names and other issues ([@JesseFarebro](https://github.com/JesseFarebro))
* ``pip install gym[atari]`` no longer distributes Atari ROMs that the ALE (the Atari emulator used) needs to run the various games. The easiest way to install ROMs into the ALE has been to use [AutoROM](https://github.com/Farama-Foundation/AutoROM). Gym now has a hook to AutoROM for easier CI automation so that using ``pip install gym[accept-rom-license]`` calls AutoROM to add ROMs to the ALE. You can install the entire suite with the shorthand ``gym[atari, accept-rom-license]``. Note that as described in the name name, by installing ``gym[accept-rom-license]`` you are confirming that you have the relevant license to install the ROMs. ([@JesseFarebro](https://github.com/JesseFarebro))
* An accidental breaking change when loading saved policies trained on old versions of Gym with environments using the box action space have been fixed. ([@RedTachyon](https://github.com/RedTachyon))
* Pendulum has had a minor fix to it's physics logic made and the version has been bumped to v1 ([@RedTachyon](https://github.com/RedTachyon))
* Tests have been refactored into an orderly manner ([@RedTachyon](https://github.com/RedTachyon))
* Dict spaces now have standard dict helper methods ([@Rohan138](https://github.com/Rohan138))
* Environment properties are now forwarded to the wrapper ([@tristandeleu](https://github.com/tristandeleu))
* Gym now properly enforces calling reset before stepping for the first time ([@ahmedo42](https://github.com/ahmedo42))
* Proper piping of error messages to stderr ([@XuehaiPan](https://github.com/XuehaiPan))
* Fix video saving issues ([@zlig](https://github.com/zlig))

Also, Gym is compiling a list of third party environments to into the new documentation website we're working on. Please submit PRs for ones that are missing: https://github.com/openai/gym/blob/master/docs/third_party_environments.md

**Full Changelog**: https://github.com/openai/gym/compare/v0.20.0...v0.21.0

**Github Release**: https://github.com/openai/gym/releases/tag/v0.21.0
