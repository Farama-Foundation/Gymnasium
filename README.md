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

## Contributing

We welcome contributions from the community!
Please see our [CONTRIBUTING.md](https://github.com/Farama-Foundation/Gymnasium/blob/main/CONTRIBUTING.md) for details on how to get started.

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

<h3 style="margin-bottom:10;margin-top:0"><a href="https://ref.wisprflow.ai/UnmiceG">Wispr Flow</a></h3>

<a href="https://ref.wisprflow.ai/UnmiceG">
  <img src="assets/wispr-flow.svg" alt="Wispr Flow" width="100">
</a>

<h3 style="margin-bottom:10;margin-top:0">Dictation that understands code</h3>
<h4 style="margin-top:0;">Ship 4x faster with developer-first dictation that works in every app.</h4>

<p style="margin-top:50;">If you'd like to sponsor Gymnasium or other Farama repositories and have your logo here, <a href="mailto:contact@farama.org">contact us</a>.</p>
