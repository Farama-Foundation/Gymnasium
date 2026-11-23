---
title: Atari
firstpage:
lastpage:
---

# Atari (moved to the Arcade Learning Environment)

```{warning}
The Atari environments and their documentation have moved to the
[Arcade Learning Environment (ALE)](https://ale.farama.org/). Gymnasium no longer
documents them here; every page below links to its new home on
[ale.farama.org](https://ale.farama.org/environments/).
```

The ALE is the Atari 2600 emulator these environments have always been built on. It
exposes them through the Gymnasium API itself, so they are created with `gymnasium.make`
exactly as before, under `ALE/` ids.

## Migration Guide

Install `ale-py`, either directly or through Gymnasium's `atari` extra:

```bash
pip install ale-py
# or
pip install "gymnasium[atari]"
```

Then register the environments with Gymnasium and make them as usual:

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

The common Atari observation pipeline stays in Gymnasium as
[`gymnasium.wrappers.AtariPreprocessing`](../api/wrappers/misc_wrappers).

## Environments

The full list of games, with per-game documentation, is at
[ale.farama.org/environments](https://ale.farama.org/environments/).

## Links

* [Documentation](https://ale.farama.org/)
* [GitHub](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
* [PyPI](https://pypi.org/project/ale-py/)
