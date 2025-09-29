[![Python](https://img.shields.io/pypi/pyversions/gymnasium.svg)](https://badge.fury.io/py/gymnasium)
[![PyPI](https://badge.fury.io/py/gymnasium.svg)](https://badge.fury.io/py/gymnasium)
[![arXiv](https://img.shields.io/badge/arXiv-2407.17032-b31b1b.svg)](https://arxiv.org/abs/2407.17032)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/github/license/Farama-Foundation/Gymnasium)](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <a href="https://gymnasium.farama.org/" target = "_blank">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium-text.png" width="500px" />
</a>

</p>

Gymnasium is an open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API. This is a fork of OpenAI's [Gym](https://github.com/openai/gym) library by its maintainers (OpenAI handed over maintenance a few years ago to an outside team), and is where future maintenance will occur going forward.

The documentation website is at [gymnasium.farama.org](https://gymnasium.farama.org), and we have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6

## Environments

Gymnasium includes the following families of environments along with a wide variety of third-party environments
* [Classic Control](https://gymnasium.farama.org/environments/classic_control/) - These are classic reinforcement learning based on real-world problems and physics.
* [Box2D](https://gymnasium.farama.org/environments/box2d/) - These environments all involve toy games based around physics control, using box2d based physics and PyGame-based rendering
* [Toy Text](https://gymnasium.farama.org/environments/toy_text/) - These environments are designed to be extremely simple, with small discrete state and action spaces, and hence easy to learn. As a result, they are suitable for debugging implementations of reinforcement learning algorithms.
* [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) - A physics engine based environments with multi-joint control which are more complex than the Box2D environments.
* [Atari](https://ale.farama.org/) - Emulator of Atari 2600 ROMs simulated that have a high range of complexity for agents to learn.
* [Third-party](https://gymnasium.farama.org/environments/third_party_environments/) - A number of environments have been created that are compatible with the Gymnasium API. Be aware of the version that the software was created for and use the `apply_env_compatibility` in `gymnasium.make` if necessary.

## Installation

To install the base Gymnasium library, use `pip install gymnasium`

This does not include dependencies for all families of environments (there's a massive number, and some can be problematic to install on certain systems). You can install these dependencies for one family like `pip install "gymnasium[atari]"` or use `pip install "gymnasium[all]"` to install all dependencies.

We support and test for Python 3.10, 3.11, 3.12 and 3.13 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.

## API

The Gymnasium API models environments as simple Python `env` classes. Creating environment instances and interacting with them is very simple- here's an example using the "CartPole-v1" environment:

```python
import gymnasium as gym
env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

## Notable Related Libraries

Please note that this is an incomplete list, and just includes libraries that the maintainers most commonly point newcomers to when asked for recommendations.

* [CleanRL](https://github.com/vwxyzjn/cleanrl) is a learning library based on the Gymnasium API. It is designed to cater to newer people in the field and provides very good reference implementations.
* [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) is a multi-agent version of Gymnasium with a number of implemented environments, i.e. multi-agent Atari environments.
* The Farama Foundation also has a collection of many other [environments](https://farama.org/projects) that are maintained by the same team as Gymnasium and use the Gymnasium API.

## Environment Versioning

Gymnasium keeps strict versioning for reproducibility reasons. All environments end in a suffix like "-v0".  When changes are made to environments that might impact learning results, the number is increased by one to prevent potential confusion. These were inherited from Gym.

## Support Gymnasium's Development

If you are financially able to do so and would like to support the development of Gymnasium, please join others in the community in [donating to us](https://github.com/sponsors/Farama-Foundation).

## Citation

You can cite Gymnasium using our related paper (https://arxiv.org/abs/2407.17032) as:

```
@article{towers2024gymnasium,
  title={Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author={Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan and Balis, John U and De Cola, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and Krimmel, Markus and KG, Arjun and others},
  journal={arXiv preprint arXiv:2407.17032},
  year={2024}
}
```

## Repository Sponsors

[<svg width="100" height="100" viewBox="0 0 187 187" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect y="-0.00576019" width="187" height="187" rx="40" fill="#1A1A1A"/>
<path d="M48.1303 41.598C46.1328 41.598 44.2093 42.3748 42.8037 43.7804C41.361 45.1861 40.5842 47.1096 40.5842 49.0701V137.848C40.5842 138.847 40.7692 139.808 41.1391 140.733C41.509 141.658 42.0639 142.472 42.7667 143.174C43.4695 143.877 44.2833 144.432 45.2081 144.802C46.1328 145.172 47.0946 145.357 48.0933 145.357C50.0908 145.357 52.0144 144.58 53.42 143.174C54.8257 141.769 55.6395 139.882 55.6395 137.885V49.0701C55.6395 47.0726 54.8626 45.1861 53.42 43.7804C52.0144 42.3748 50.0908 41.598 48.0933 41.598H48.1303Z" fill="#FFFFEB"/>
<path d="M93.4069 58.5766C92.4081 58.5766 91.4464 58.7615 90.5216 59.1314C89.5968 59.5014 88.783 60.0562 88.0802 60.759C87.3774 61.4619 86.8225 62.2757 86.4526 63.2004C86.0827 64.1252 85.8608 65.0869 85.8608 66.0487V120.166C85.8608 122.164 86.6376 124.05 88.0802 125.456C89.4859 126.861 91.4094 127.638 93.4069 127.638C95.4044 127.638 97.3279 126.861 98.7336 125.456C100.139 124.05 100.953 122.164 100.953 120.166V66.0487C100.953 64.0512 100.176 62.1647 98.7336 60.759C97.3279 59.3534 95.4044 58.5766 93.4069 58.5766Z" fill="#FFFFEB"/>
<path d="M70.7647 89.1309C68.7672 89.1309 66.8437 89.9077 65.4381 91.3134C64.0324 92.719 63.2186 94.6056 63.2186 96.6031V127.453C63.2186 129.451 63.9954 131.337 65.4381 132.743C66.8437 134.149 68.7672 134.925 70.7647 134.925C72.7622 134.925 74.6858 134.149 76.0914 132.743C77.497 131.337 78.3108 129.451 78.3108 127.453V96.6031C78.3108 94.6056 77.534 92.719 76.0914 91.3134C74.6858 89.9077 72.7622 89.1309 70.7647 89.1309Z" fill="#FFFFEB"/>
<path d="M138.686 41.598C137.687 41.598 136.725 41.7829 135.8 42.1528C134.876 42.5227 134.062 43.0776 133.359 43.7804C132.656 44.4832 132.101 45.297 131.731 46.2218C131.361 47.1466 131.14 48.1083 131.14 49.0701V137.848C131.14 139.845 131.916 141.732 133.359 143.137C134.765 144.543 136.688 145.32 138.686 145.32C140.683 145.32 142.607 144.543 144.012 143.137C145.418 141.732 146.232 139.845 146.232 137.848V49.0701C146.232 47.0726 145.455 45.1861 144.012 43.7804C142.607 42.3748 140.683 41.598 138.686 41.598Z" fill="#FFFFEB"/>
<path d="M116.043 89.1309C114.046 89.1309 112.122 89.9077 110.717 91.3134C109.311 92.719 108.497 94.6056 108.497 96.6031V127.453C108.497 129.451 109.274 131.337 110.717 132.743C112.122 134.149 114.046 134.925 116.043 134.925C118.041 134.925 119.964 134.149 121.37 132.743C122.776 131.337 123.59 129.451 123.59 127.453V96.6031C123.59 94.6056 122.813 92.719 121.37 91.3134C119.964 89.9077 118.041 89.1309 116.043 89.1309Z" fill="#FFFFEB"/>
</svg>](https://ref.wisprflow.ai/UnmiceG)

<h3 style="margin-bottom:10;margin-top:0">Dictation that understands code</h3>
<h4 style="margin-top:0;">Ship 4x faster with developer-first dictation that works in every app.</h4>

<p style="margin-top:50;">If you'd like to sponsor Gymnasium or other Farama repositories and have your logo here, <a href="mailto:contact@farama.org">contact us</a>.</p>
