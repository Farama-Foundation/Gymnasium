---
layout: "contents"
title: v26 step API
---

## TL;DR 

In gym versions prior to v0.25, the step API was
```python
>>> obs, reward, done, info = env.step(action)
```

In https://github.com/openai/gym/pull/2752, the step API was changed to
```python
>>> obs, reward, terminated, truncated, info = env.step(action)
>>> done = terminated or truncated
```
`terminated` dictates when the environment state has entered a terminal state, whereas `truncated` was used to indicate that the environment enters a truncated state (i.e., reaching a time limit, moving out of bounds) . When training agent should use the `terminated` for if bootstrap should happen, whereas `truncated` should not be used when looping through the environment.

All environments and wrappers from v0.26 should support this new API. For environments that have not updated, we provide built in conversion for environment in v0.22 style API to the v0.26 API using the `EnvCompatibility` wrapper, and via make:
```python
gym.make(..., apply_api_compatibility=True)
```

In Gym v0.25, these changes are already included but are turned off by default. Therefore, we encourage all users to update to v0.26, or alternatively use v0.23.1 that does not include these changes for existing experiments.

# (Long Explanation) Terminated / Truncated Step API

In this post, we explain the motivation for the change, what the new `Env.step` API is, why alternative implementations were not selected, and finally the suggested code changes for developers.  

## Introduction

In most environments, it is possible for an agent to take a series of cyclical actions, resulting in what is effectively wandering in circles. As long as it does not encounter a terminal state, it possible for environments to never end. To prevent this phenomenon, Gym lets environments have the option to specify a time limit that the agent must complete the environment within, after which the environment is considered to be **truncated**. Only in cases where an agent stumbles upon a terminal state (e.g. falling off a cliff), is the environment considered to be **terminated**. In most cases, the time limit and current timestep is unbeknownst to the agent, being absent from its observations. This results in a "non-Markovian end condition" from the agent’s perspective. This is an issue if we treat ending via time limit and ending via terminal state as equal as there is no way for an agent to know whether the environment has terminated or simply been truncated.

This matters for Reinforcement Learning algorithms that perform bootstrapping to update value estimates (e.g. Value function, Q-value) present in all popular algorithms such as DQN, A2C, SAC, PPO, etc [[1]](https://arxiv.org/pdf/1712.00378.pdf). To illustrate why, we refer to Algorithm 1 (Page 5) of the original [DQN paper](https://arxiv.org/abs/1312.5602). In the following example for updating the Q-value, the next Q-value depends on if the environment has terminated.

```
If terminated:  # case 1
    Next q-value = reward
Else:  # case 2
    Next q-value = reward + discount factor * max action of the Q (next state, action)
 
# This can more efficiently be written
Next q-value = reward + (not terminated) * discount factor * max action of the Q(next state, action)
```

In the case when the environment has simply truncated, `case 2` should still be applied. Unfortunately, it is impossible to tell whether an environment has terminated or simply truncated when both signals are aggregated into a single `done` flag for an environment. This data was included in the step `info[TimeLimit.truncated]` parameter however in surveying open source reinforcement learning implementations we found an alarming amount were not aware of this additional info, necessary for training. **This was the main motivation for changing the step API; which we aim to encourage more accurate implementations of reinforcement learning algorithms, a critical factor in academia when replicating work.**

The reason that most users are unaware of the difference between truncation and termination is that documentation on this issue was missing. This can be seen in the top 4 tutorials found searching google for “DQN tutorial”, [[1]](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), [[2]](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc), [[3]](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial), [[4]](http://seba1511.net/tutorials/intermediate/reinforcement_q_learning.html) (checked 21 July 2022). Only a single website (Tensorflow Agents) implements truncation and termination correctly. Importantly, the reason that Tensorflow Agent does not fall for this issue is that Google has recognised this issue with the Gym `step` implementation and has designed their own API where the `step` function returns the `discount factor` instead of `done`. [See time step codeblock](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial#environment). 

## New Terminated / Truncated Step API

In this Section, we discuss the terminated / truncated step API along with the changes made to Gym that will affect users. We should note that these changes might not be implemented by all python modules or tutorials that use Gym. In v0.25, this behavior will be turned off by default (in a majority of cases) but after v0.26, support for the old step API is provided solely through the `EnvCompatibility` and `StepAPICompatibility` wrapper. 

1. **Core API change** - All environments, wrappers, utils and vector implementations within Gym (i.e., CartPole) have been changed to the new API.
```python
# (old) done step API
def step(self, action) -> Tuple[ObsType, float, bool, dict]:
# (new) terminated / truncated step API
def step(self, action) -> Tuple[ObsType, float, bool, bool, dict]:
```

2. **Changes in phrasing** - In the vector environments, `terminal_reward`, `terminal_observation` etc. is replaced with `final_reward`, `final_observation` etc. The intention is to reserve the 'termination' word for only if `terminated=True`. For some motivation, Sutton and Barto use terminal states to specifically refer to special states whose values are 0, states at the end of the MDP. This is not true for a truncation where the value of the final state need not be 0. So the current usage of `terminal_obs` etc. would be incorrect if we adopt this definition.

## Suggested Code changes

We believe there are primarily two changes that will have to be made by developers updating to the new Step API.

1. **Stepping through the environment** - You need to change the `env.step` to take 5 elements, `obs, reward, termination, truncation, info = env.step(action)`. To loop through the environment then you need to check if the environment needs resetting with `done = terminated or truncated`. 
```python
env = gym.make(“CartPole-v1”)
done = False
while not done:
	action = env.action_space.sample()
	obs, reward, terminated, truncated, info = env.step(action)
	done = terminated or truncated
```

2. **Training of Agents** - As explained before, differentiating between termination and truncation is critical for the training of bootstrap-based agents. You could get `terminated` and `truncated` from `info["TimeLimit.truncated"]` in the old step API to correctly implement many RL algorithms. In the new API, usage of `terminated` and `truncated` is unique for each algorithm implementation, but generally, only the `termination` information is used for evaluating bootstrapped estimates, thus, replay buffers and episode storage can generally replace `done` with `terminated` safely, while `terminated or truncated` can be used to determine when to reset.

## Backward compatibility

To allow conversions between the old and new step APIs, we provide `convert_to_terminated_truncated_step_api` and `convert_to_done_step_api` in `utils/step_api_compatibility.py`. These functions are also incorporated within the `StepAPICompatibility` and `EnvCompatibility` wrappers.

For users that wish to convert between the two API, these functions can be used between environments, wrappers, vectorisation and outside code. Example usage, 
```python
# wrapper's step method
def step(self, action):
    # here self.env.step is made to return in new API, since the wrapper is written in new API
    obs, reward, done, info = convert_to_done_step_api(self.env.step(action)) 
    if done:
        ### terminated / truncated code
    ### more wrapper code
    # here the wrapper is made to return in API specified by convert_to_terminated_truncated_step_api
    return convert_to_terminated_truncated_step_api((obs, reward, done, info)) 
```

With the step compatibility functions, whenever an environment (or sub-environment with vectorisation) is terminated or truncated, `"TimeLimit.truncated"` is added to the step `info`. However, if the environment `terminated=True` and `truncated=True` in the same timestep, `info["TimeLimit.truncated"]` evaluates to `False`. 

## Alternative Implementations

While developing this new Step API, a number of developers asked why alternative implementations were not taken. 
There are four primary alternative approaches that we considered:

* No change: With changes to the documentation alone, it is possible for developers to accurately implement Reinforcement Learning algorithms with termination and truncation. However, due to the prevalence of this misunderstanding within the Reinforcement Learning community (as shown in the short survey of tutorials above), we are skeptical that changes in documentation and a blog post will change the course of this misunderstanding. Therefore, we believe no change would not cause the community to fix the root issue. 
* Custom Boolean: It is feasible to replace `done` which is a python bool with a custom bool implementation that can act identically to boolean except in addition encoding the `truncation` information. Similar to this is a proposal to replace `done` as an integer to allow the four possible `termination` and `truncation` states. However, the primary problem here is that significant bugs as it is not obvious what step API an environment might be using.
* Discount factor: For [Deepmind Env](https://github.com/deepmind/dm_env/blob/master/docs/index.md) and [TensorflowAgent](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial#environment), the `step` function returns the `discount_factor` instead of `done`. This allows them to have variable `discount_factors` over an episode, and a discount factor of `0` for the terminal step. Similar to the custom boolean proposal, we believe that this could just introduce more bugs with confusion between the old and new APIs. In addition, Gym was designed to be agnostic to methods of solving the environments, and the discounted future reward paradigm is but one of many methods that practitioners may view as a reinforcement learning problem.
* 5 elements: While we agree that our proposed 5-element tuple is not optimal, we believe our proposal is the best for the future. One of the primary reasons is that the change makes assessing if code follows the new or old API easy and avoids the issue of being partially backward compatible, preventing many problematic reproducibility edge cases when both APIs are used at the same time.

## Related Reinforcement Learning Theory

Reinforcement Learning tasks can be grouped into two camps - episodic tasks and continuing tasks. Episodic tasks refer to environments that terminate in a finite number of steps. This can further be divided into Finite-Horizon tasks which end in a *fixed* number of steps and Indefinite Horizon tasks which terminate in an arbitrary number of steps but must end (e.g. goal completion, task failure). In comparison, Continuing tasks refer to environments which have *no* end (eg. some control process tasks). 

The state that leads to an episode ending in episodic tasks is referred to as a terminal state, and the value function of this state is 0. The episode is said to have terminated when the agent reaches this state. All this is encapsulated within the Markov Decision Process (MDP) which defines a task or environment.

A critical difference occurs in practice when we choose to end the episode for reasons outside the scope of the agent. This is typically in the form of time limits set to limit the number of timesteps per episode (useful for several reasons - batching, diversifying experience etc.). This kind of truncation is essential in training continuing tasks that have no end, but also useful in episodic tasks that can take an arbitrary number of steps to end. This condition can also be in the form of an out-of-bounds limit, where the episode ends if a robot steps out of a boundary, but this is more due to a physical restriction and not part of the task itself. 

We can thus differentiate the reason for an episode ending into two categories - the agent reaching a terminal state as defined under the MDP of the task, and the agent satisfying a condition that is out of the scope of the MDP. We refer to the former condition as termination and the latter condition as truncation. 

Note that while finite horizon tasks end due to a time limit, this would be considered a termination since the time limit is built into the task. For these tasks, to preserve the Markov property, it is essential to add information about ‘time remaining’ in the state. For this reason, Gym includes a `TimeObservation` wrapper for users who wish to include the current time step in the agent’s observation.

