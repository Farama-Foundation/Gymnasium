---
layout: "contents"
title: Recording Agents
---

# Recording Agents

```{eval-rst}
.. py:currentmodule: gymnasium.wrappers

During training or when evaluating an agent, it may be interesting to record agent behaviour over an episode and log the total reward accumulated. This can be achieved through two wrappers: :class:`RecordEpisodeStatistics` and :class:`RecordVideo`, the first tracks episode data such as the total rewards, episode length and time taken and the second generates mp4 videos of the agents using the environment renderings.

We show how to apply these wrappers for two types of problems; the first for recording data for every episode (normally evaluation) and second for recording data periodiclly (for normal training).
```

## Recording Every Episode

```{eval-rst}
.. py:currentmodule: gymnasium.wrappers

Given a trained agent, you may wish to record several episodes during evaluation to see how the agent acts. Below we provide an example script to do this with the :class:`RecordEpisodeStatistics` and :class:`RecordVideo`.
```

```python
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 4

env = gym.make("CartPole-v1", render_mode="rgb_array")  # replace with your environment
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
                  episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # replace with actual agent
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
env.close()

print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')
```

```{eval-rst}
.. py:currentmodule: gymnasium.wrappers

In the script above, for the :class:`RecordVideo` wrapper, we specify three different variables: ``video_folder`` to specify the folder that the videos should be saved (change for your problem), ``name_prefix`` for the prefix of videos themselves and finally an ``episode_trigger`` such that every episode is recorded. This means that for every episode of the environment, a video will be recorded and saved in the style "cartpole-agent/eval-episode-x.mp4".

For the :class:`RecordEpisodicStatistics`, we only need to specify the buffer lengths, this is the max length of the internal ``time_queue``, ``return_queue`` and ``length_queue``. Rather than collect the data for each episode individually, we can use the data queues to print the information at the end of the evaluation.

For speed ups in evaluating environments, it is possible to implement this with vector environments to in order to evaluate ``N`` episodes at the same time in parallel rather than series.
```

## Recording the Agent during Training

During training, an agent will act in hundreds or thousands of episodes, therefore, you can't record a video for each episode, but developers might still want to know how the agent acts at different points in the training, recording episodes periodically during training. While for the episode statistics, it is more helpful to know this data for every episode. The following script provides an example of how to periodically record episodes of an agent while recording every episode's statistics (we use the python's logger but [tensorboard](https://www.tensorflow.org/tensorboard), [wandb](https://docs.wandb.ai/guides/track) and other modules are available).

```python
import logging

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

training_period = 250  # record the agent's episode every 250
num_training_episodes = 10_000  # total number of training episodes

env = gym.make("CartPole-v1", render_mode="rgb_array")  # replace with your environment
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)
env = RecordEpisodeStatistics(env)

for episode_num in range(num_training_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # replace with actual agent
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    logging.info(f"episode-{episode_num}", info["episode"])
env.close()
```

## More information

* [Training an agent](train_agent.md)
* [More training tutorials](../tutorials/training_agents)
