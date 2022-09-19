---
title: Vector
---

# Vector

```{eval-rst}
.. autofunction:: gymnasium.vector.make
```  


## VectorEnv

```{eval-rst}
.. attribute:: gymnasium.vector.VectorEnv.action_space

    The (batched) action space. The input actions of `step` must be valid elements of `action_space`.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.action_space
        MultiDiscrete([2 2 2])

.. attribute:: gymnasium.vector.VectorEnv.observation_space

    The (batched) observation space. The observations returned by `reset` and `step` are valid elements of `observation_space`.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.observation_space
        Box([[-4.8 ...]], [[4.8 ...]], (3, 4), float32)

.. attribute:: gymnasium.vector.VectorEnv.single_action_space

    The action space of an environment copy.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.single_action_space
        Discrete(2)

.. attribute:: gymnasium.vector.VectorEnv.single_observation_space

    The observation space of an environment copy.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.single_action_space
        Box([-4.8 ...], [4.8 ...], (4,), float32)
``` 



### Reset

```{eval-rst}
.. automethod:: gymnasium.vector.VectorEnv.reset
``` 

```python
>>> import gymnasium as gym
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.reset()
(array([[-0.02240574, -0.03439831, -0.03904812,  0.02810693],
       [ 0.01586068,  0.01929009,  0.02394426,  0.04016077],
       [-0.01314174,  0.03893502, -0.02400815,  0.0038326 ]],
      dtype=float32), {})
```
### Step

```{eval-rst}
.. automethod:: gymnasium.vector.VectorEnv.step
``` 

```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.reset()
>>> actions = np.array([1, 0, 1])
>>> observations, rewards, terminated, truncated, infos = envs.step(actions)

>>> observations
array([[ 0.00122802,  0.16228443,  0.02521779, -0.23700266],
        [ 0.00788269, -0.17490888,  0.03393489,  0.31735462],
        [ 0.04918966,  0.19421194,  0.02938497, -0.29495203]],
        dtype=float32)
>>> rewards
array([1., 1., 1.])
>>> terminated
array([False, False, False])
>>> infos
{}
```
