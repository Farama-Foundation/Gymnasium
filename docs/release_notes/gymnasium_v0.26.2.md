---
layout: "contents"
title: Gymnasium v0.26.2
---

# v0.26.2 Release Notes

This Release is an upstreamed version of [Gym v0.26.2](https://github.com/openai/gym/releases/tag/0.26.2)

### Bugs Fixes
* As reset now returns (obs, info) then in the vector environments, this caused the final step's info to be overwritten. Now, the final observation and info are contained within the info as "final_observation" and "final_info" [@pseudo-rnd-thoughts](https://github.com/pseudo-rnd-thoughts)
* Adds warnings when trying to render without specifying the render_mode [@younik](https://github.com/younik)
* Updates Atari Preprocessing such that the wrapper can be pickled [@vermouth1992](https://github.com/vermouth1992)
* Github CI was hardened to such that the CI just has read permissions [@sashashura](https://github.com/sashashura)
* Clarify and fix typo in GraphInstance [@ekalosak](https://github.com/ekalosak)

**Github Release**: https://github.com/Farama-Foundation/Gymnasium/releases/tag/v0.26.2
