---
layout: "contents"
title: v26 step API
---

## TL;DR 

In https://github.com/openai/gym/pull/2752, we have recently changed the Gym `Env.step` API.

In gym versions prior to v25, the step API was
```python
>>> obs, reward, done, info = env.step(action)
```
In gym versions 26, the step API was changed to
```python
>>> obs, reward, terminated, truncated, info = env.step(action)
>>> done = terminated or truncated
```
For training agents, where `done` was used previously, then `terminated` should be used. 

In v26, all internal gym environments and wrappers solely support the (new) terminated / truncated step API with support
for the (old) done step API being provided through the `EnvCompatibility` wrapper for converting between the old and 
new APIs, accessible through `gym.make(..., apply_api_compatibility=True)`.

It should be noted that v25 has the changes listed below but is turned off by default and require parameters not 
discussed in this blog post. Therefore, we recommend users to either update to v26 or use v23.1 that do not include
these changes. 

For a detailed explanation of the changes and reasoning, read the rest of this post.

# (New) Terminated / Truncated Step API

In this post, we explain the motivation for the change, what the new `Env.step` API is, why alternative implementations
were not selected and the suggested code changes for developers.  

## Introduction

To prevent an agent from wandering in circles forever, not doing anything, and for other practical reasons, 
Gym lets environments have the option to specify a time limit that the agent must complete the environment within. 
Importantly, this time limit is outside of the agent’s knowledge as it is not contained within their observations. 
Therefore, when the agent reaches the time limit, the environment should be reset however **this type of reset should 
be treated differently from when the agent reaches a goal and the environment ends**. We refer to the first type as 
**truncation**, when the agent reaches the time limit (maximum number of steps) for the environment, and the second 
type as **termination**, when the environment state reaches a specific condition (i.e. the agent reaches the goal). 
For a more precise discussion of how Gym works in relation to RL theory, see the [theory](#theory) section. 

The problem is that **most users of Gym have treated termination and truncation as identical**. 
Gym's step API `done` signal only referred to the fact that the environment needed resetting with `info`, 
`“TimeLimit.truncation”=True or False` specifying if the cause is `truncation` or `termination`. 

This matters for most Reinforcement Learning algorithms [[1]](https://arxiv.org/pdf/1712.00378.pdf) that perform 
bootstrapping to update the Value function or related estimates (i.e. Q-value), used by DQN, A2C, etc. 
In the following example for updating the Q-value, the next Q-value depends on if the environment has terminated. 

```
If terminated:  # case 1
    Next q-value = reward
Else:  # case 2
    Next q-value = reward + discount factor * max action of the Q (next state, action)

# This can more efficiently be written
Next q-value = reward + (not terminated) * discount factor * max action of the Q(next state, action)
```

This can be seen in Algorithm 1 (Page 5) of the original [DQN paper](https://arxiv.org/abs/1312.5602), however, we noted
that this case is often ignored when writing the pseudocode for Reinforcement Learning algorithms. 

Therefore, if the environment has truncated and not terminated, case 2 of the bootstrapping should be computed, however,
if the case is determined by `done`, this can result in the wrong implementation. **This was the main motivation for 
changing the step API to encourage accurate implementations, a critical factor for academia when replicating work.**

The reason that most users are unaware of this difference between truncation and termination is that documentation on
this issue was missing. As a result, a large amount of tutorial code has incorrectly implemented RL algorithms. 
This can be seen in the top 4 tutorials found searching google for “DQN tutorial”, 
[[1]](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), 
[[2]](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc), 
[[3]](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial), 
[[4]](http://seba1511.net/tutorials/intermediate/reinforcement_q_learning.html) (checked 21 July 2022) where only a 
single website (Tensorflow Agents) implements truncation and termination correctly. Importantly, the reason that 
Tensorflow Agent does not fall for this issue is that Google has recognised this issue with the Gym `step` 
implementation and has designed their own API where the `step` function returns the `discount factor` instead of `done`.
[See time step code block](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial#environment). 

## (New) Terminated / Truncated Step API

In this Section, we discuss the (new) terminated / truncated step API along with the changes made to Gym that will 
affect users. We should note that these changes might not be implemented by all python modules or tutorials that use Gym. 
In `v0.25`, this behaviour will be turned off by default (in a majority of cases) but in `v0.26+`, support for the old 
step API is provided solely through the `EnvCompatibility` and `StepAPICompatibility` wrapper. 

1. All environments, wrappers, utils and vector implementations within Gym (i.e., CartPole) have been changed to the new API.
  Warnings, this might not be true outside of gym.
```python
# (old) done step API
def step(self, action) -> Tuple[ObsType, float, bool, dict]:

# (new) terminated / truncated step API
def step(self, action) -> Tuple[ObsType, float, bool, bool, dict]:
```

2. Changes in phrasing - In the vector environments, `terminal_reward`, `terminal_observation` etc. is replaced with 
  `final_reward`, `final_observation` etc. The intention is to reserve the 'termination' word for only if
  `terminated=True`. (for some motivation, Sutton and Barto use terminal states to specifically refer to special 
  states whose values are 0, states at the end of the MDP. This is not true for a truncation where the value of the 
  final state need not be 0. So the current usage of `terminal_obs` etc. would be incorrect if we adopt this definition)

## Suggested Code changes

We believe there are primarily two changes that will have to be made by developers updating to the new Step API.

1. Stepping through the environment - You need to change the `env.step` to take 5 elements, 
  `obs, reward, termination, truncation, info = env.step(action)`. To loop through the environment then you need to 
  check if the environment needs resetting with `done = terminated or truncated`. 
```python
env = gym.make(“CartPole-v1”)
done = False
while not done:
	action = env.action_space.sample()
	obs, reward, terminated, truncated, info = env.step(action)
	done = terminated or truncated
```

2. Training of Agents - As explained before, differentiating between termination and truncation is critical for the
  training of bootstrap-based agents. Using the (old) done step API required determining `terminated` and `truncated` 
  from `info["TimeLimit.truncated"]` to correctly implement many RL algorithms. We should note that it is not possible 
  for both `terminated` and `truncated` to be both true with the (old) done step API which is possible for the new API. 
  For the (new) terminated / truncated step API, `terminated` and `truncated` is known immediately from `env.step`. 
  To use `terminated` and `truncated` is unique for each algorithms implementation but the `termination` information is
  generally critical for bootstrapped estimated training algorithms and in replay buffers can generally replace `done`. 
  However, check if the training code has been updated.

### Backward compatibility

To allow conversions between the done step API and termination / truncation step API, we provide 
`convert_to_terminated_truncated_step_api` and `convert_to_done_step_api` in `utils/step_api_compatibility.py`. 
These functions work with vector environments (with both list and dictionary-based info). 
These functions are incorporated with the `StepAPICompatibility` and `EnvCompatibility` wrappers. 

This function is similar to the wrapper, it is used for backward compatibility in wrappers, vector envs. 
It is used at interfaces between environments, wrappers, vectorisation and outside code. Example usage, 
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

With the step compatibility functions, whenever an environment (or sub-environment with vectorisation) is terminated 
or truncated, `"TimeLimit.truncated"` is added to the step `info`. However, as the info cannot specify if 
`terminated` and `truncated` is True only one being True, in cases of converting, `termination` is favored over 
`truncation`. I.e. if `terminated=True` and `truncated=True` then `done=True` and `info['TimeLimit.truncated']=False`. 
The revert is also assumed if `done=True` and `info["TimeLimit.truncated"]=True`, then `terminated=False` and `truncated=True`. 

## Alternative Implementations

While developing this new Step API, a number of developers asked why alternative implementations were not taken. 
There are four primary alternative approaches that we considered:

* No change: With changes to the documentation alone, it is possible for developers to accurately implement 
  Reinforcement Learning algorithms with termination and truncation. However, due to the prevalence of this 
  misunderstanding within the Reinforcement Learning community (as shown in the short survey of tutorials at the end 
  of the introduction), we are sceptical that solely creating documentation and some blog posts would cause a 
  significant shift within the community to fix the issue in the long term. Therefore, we believe no change would not 
  cause the community to fix the root issue. 
* Custom Boolean: It is feasible to replace `done` which is a python bool with a custom bool implementation that can act 
  identically to boolean except in addition encoding the `truncation` information. Similar to this is a proposal to 
  replace `done` as an integer to allow the four possible `termination` and `truncation` states. However, the primary 
  problem with both of these implementations is that it is backwards compatible meaning that (old) done code that is not
  properly implemented with new custom boolean or integer step API could cause significant bugs to occur. As a result, 
  we believe this proposal could cause significantly more issues. 
* Discount factor: For [Deepmind Env](https://github.com/deepmind/dm_env/blob/master/docs/index.md) and 
  [TensorflowAgent](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial#environment), the `step` function 
  return the `discount_factor` instead of `done`. This allows them to have variable `discount_factors` over an episode
  and can address the issue with termination and truncation. However, we identify two problems with this proposal. 
  The first is similar to the custom boolean implementation that while the change is backwards compatible this can 
  instead cause assessing if the tutorial code is updated to the new API. The second issue is that Gym provides an API
  solely for environments, and is agnostic to the solving method. So adding the discount factor would change one of the
  core Gym philosophies.
* 5 elements: While we agree that our proposed 5-element tuple is not optimal (there are many things like the step API 
  in Gym which if developed in 2022 with the goal of making a de facto standard for Reinforcement Learning, we would 
  certainly change), we believe our proposal is the best for the future. One of the primary reasons is that the change
  makes assessing if code follows the new or old API easy and avoids the issue of being partially backward compatible 
  allowing new bugs to occur. 

## Related Reinforcement Learning Theory

Reinforcement Learning tasks into grouped into two - episodic tasks and continuing tasks. Episodic tasks refer to 
environments that terminate in a finite number of steps. This can further be divided into Finite-Horizon tasks which 
end in a *fixed* number of steps and Indefinite Horizon tasks which terminate in an arbitrary number of steps but must 
end (eg. goal completion, task failure). In comparison, Continuing tasks refer to environments which have *no* end 
(eg. some control process tasks). 

The state that leads to an episode ending in episodic tasks is referred to as a terminal state, and the value of this 
state is 0. The episode is said to have terminated when the agent reaches this state. All this is encapsulated within 
the Markov Decision Process (MDP) which defines a task (Environment). 

A critical difference occurs in practice when we choose to end the episode for reasons outside the scope of the agent 
(MDP). This is typically in the form of time limits set to limit the number of timesteps per episode 
(useful for several reasons - batching, diversifying experience etc.). This kind of truncation is essential in training
continuing tasks that have no end, but also useful in episodic tasks that can take an arbitrary number of steps to end.
This condition can also be in the form of an out-of-bounds limit, where the episode ends if a robot steps out of a 
boundary, but this is more due to a physical restriction and not part of the task itself. 

We can thus differentiate the reason for an episode ending into two categories - the agent reaching a terminal state
as defined under the MDP of the task, and the agent satisfying a condition that is out of the scope of the MDP. 
We refer to the former condition as termination and the latter condition as truncation. 

Note that while finite horizon tasks end due to a time limit, this would be considered a termination since the time 
limit is built into the task. For these tasks, to preserve the Markov property, it is essential to add information 
about ‘time remaining’ in the state. For this reason, Gym includes a `TimeObservation` wrapper for users who wish to 
include the current time step in the agent’s observation.
