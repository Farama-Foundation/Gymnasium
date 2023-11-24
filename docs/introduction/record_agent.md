---
layout: "contents"
title: Recording Agents
---

# Recording Agents

```{eval-rst}
 .. py:currentmodule: gymnasium.wrappers

During and after training an agnt, it is interesting to record how the agent acts over an episode, in particular to video the agent behaviour and log the total reward for each episode. This can be achieved through two wrappers: :class:`RecordEpisodeStatistics` and :class:`RecordVideo`, the first will track episode data such as the total rewards, episode length and time taken with the second saving the environment rendering as mp4 videos.

We consider how to apply these wrappers for two types of problems; the first for recording data for every episode and second for recording data periodiclly.
```

## Recording Every Episode

Given a trained agent, when evaluating an agent, you would wish to record several episodes to see how the agent acts.

```python
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 4

env = gym.make("CartPole-v1")  # replace with your environment
env = RecordVideo(env, video_folder="", episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # replace with actual agent
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')
env.close()
```

TODO - add description of Recordvideo

TODO - add description of record episode statistics

## Recording the Agent during Training

During training, agent will act over hundreds or thousands, therefore, you don't want to video each episode and you will want to record the episode data to a file.

```python
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

TODO

```

TODO - add description of record video

TODO - add description of record episode statistics

## More information

* [Training an agent]()
