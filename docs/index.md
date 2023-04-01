---
hide-toc: true
firstpage:
lastpage:
---

<center>
	<h1>
		Gymnasium is a standard API for reinforcement learning, and a diverse collection of reference environments.
	</h1>
</center>

<center>
	<p>Note: The video includes clips with trained agents from Stable Baselines3. (<a href="https://huggingface.co/sb3">Link</a>)</p>
	<video autoplay loop muted inline width="450" src="_static/videos/environments-demo.mp4" type="video/mp4"></video>
</center>

Gymnasium is a maintained fork of OpenAI’s Gym library. It provides a user-friendly, pythonic interface for creating and interacting with reinforcement learning environments. With Gymnasium, you can access a diverse collection of environments, as well as represent your own custom RL environments. If you require an environment that is only available in the old Gym, you can use the [compatibility wrapper](content/gym_compatibility).

Here is a minimal code example to run an environment:

```{code-block} python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42) # get the first observation

for step in range(1000):
	# here you can use your policy to get an action based on the observation
	action = env.action_space.sample()

	# execute the action in the environment
	observation, reward, terminated, truncated, info = env.step(action)

	if terminated or truncated:
		observation, info = env.reset()
env.close()
```

```{toctree}
:hidden:
:caption: Introduction

content/basic_usage
content/gym_compatibility
content/migration-guide
```

```{toctree}
:hidden:
:caption: API

api/env
api/registry
api/spaces
api/wrappers
api/vector
api/utils
api/experimental
```

```{toctree}
:hidden:
:caption: Environments

environments/classic_control
environments/box2d
environments/toy_text
environments/mujoco
environments/atari
environments/third_party_environments
```

```{toctree}
:hidden:
:glob:
:caption: Tutorials

tutorials/**/index
Comet Tutorial <https://www.comet.com/docs/v2/integrations/ml-frameworks/gymnasium/?utm_source=gymnasium&utm_medium=partner&utm_campaign=partner_gymnasium_2023&utm_content=docs_gymnasium>
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/Gymnasium>
gymnasium_release_notes/index
gym_release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/Gymnasium/blob/main/docs/README.md>
```
