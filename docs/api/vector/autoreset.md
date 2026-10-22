---
title: Autoreset Mode
---

# Autoreset Mode

When a sub-environment's episode ends (`terminated or truncated` is `True`), it must be reset before it can take further actions. Rather than wait for every sub-environment to finish, vector environments reset sub-environments individually, either automatically (autoreset) or when the user asks. The {py:class}`gymnasium.vector.AutoresetMode` decides when this happens, so it changes what {py:meth}`VectorEnv.step() <gymnasium.vector.VectorEnv.step>` returns at the end of an episode and how training code must collect transitions. There are three modes:

* {ref}`Next-step <autoreset-next-step>` (`AutoresetMode.NEXT_STEP`, the default) - the sub-environment is reset on the step after its episode ends.
* {ref}`Same-step <autoreset-same-step>` (`AutoresetMode.SAME_STEP`) - the sub-environment is reset within the step that its episode ends.
* {ref}`Disabled <autoreset-disabled>` (`AutoresetMode.DISABLED`) - the sub-environment isn't reset automatically, the user resets it with `reset(options={"reset_mask": ...})`.

The animation below follows a sub-environment whose episode ends after its third step (`s₃` is its final observation), showing the observation returned by each vector environment call in each mode.

```{image} /_static/diagrams/autoreset_modes.svg
:alt: Animated timeline of the observations returned by each call for the next-step, same-step and disabled autoreset modes
:align: center
:class: only-light
```

```{image} /_static/diagrams/autoreset_modes_dark.svg
:alt: Animated timeline of the observations returned by each call for the next-step, same-step and disabled autoreset modes
:align: center
:class: only-dark
```

A training algorithm written for one mode won't sample correctly from a vector environment using another, so check which mode an environment uses and which modes your wrappers and algorithms support.

## Choosing and reading the mode

{py:class}`SyncVectorEnv <gymnasium.vector.SyncVectorEnv>` and {py:class}`AsyncVectorEnv <gymnasium.vector.AsyncVectorEnv>` support all three modes through their `autoreset_mode` argument, which takes either an `AutoresetMode` or its string value (`"NextStep"`, `"SameStep"` or `"Disabled"`). With {py:func}`gymnasium.make_vec`, the argument is passed through `vector_kwargs`.

```python
import gymnasium as gym
from gymnasium.vector import AutoresetMode

envs = gym.make_vec(
    "CartPole-v1",
    num_envs=4,
    vectorization_mode="sync",  # or "async"
    vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
)
envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(4)],
    autoreset_mode=AutoresetMode.SAME_STEP,
)
envs = gym.vector.AsyncVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(4)],
    autoreset_mode="SameStep",
)
```

`vector_kwargs` only applies to the `"sync"` and `"async"` vectorization modes. Environments with a custom vector implementation (the `"vector_entry_point"` mode, e.g., `CartPole-v1` when `vectorization_mode` isn't given) define their own autoreset mode, and passing `vector_kwargs` to them raises an error.

Every vector environment should record its mode in `metadata["autoreset_mode"]`, which {py:func}`gymnasium.make_vec` warns about if missing, and which vector wrappers use to adapt their behaviour.

```python
>>> envs.metadata["autoreset_mode"]
<AutoresetMode.SAME_STEP: 'SameStep'>
```

(autoreset-next-step)=
## Next-step mode

On the step that a sub-environment's episode ends, `step()` returns its final observation, reward, `terminated`/`truncated` and info as normal. On the following step, the sub-environment's action is ignored and the sub-environment is reset instead, returning the reset observation and info, a reward of `0` and `terminated = truncated = False`.

The transition returned by this reset step doesn't come from the environment (it connects the final observation of one episode to the first observation of the next), so training code must skip or mask it, using the previous step's `terminated or truncated`.

```python
import gymnasium as gym
import numpy as np

envs = gym.make_vec("CartPole-v1", num_envs=4, vectorization_mode="sync")  # next-step by default
replay_buffer = []

obs, infos = envs.reset(seed=42)
autoreset = np.zeros(envs.num_envs, dtype=bool)
for _ in range(1_000):
    actions = envs.action_space.sample()  # replace with your policy, `policy(obs)`
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    for j in range(envs.num_envs):
        # skip the reset steps, `next_obs[j]` is the first observation of a new episode, not the result of `actions[j]`
        if not autoreset[j]:
            # on truncation, `next_obs[j]` is the episode's final observation to bootstrap from
            replay_buffer.append((obs[j], actions[j], rewards[j], terminations[j], next_obs[j]))

    obs = next_obs
    autoreset = np.logical_or(terminations, truncations)
envs.close()
```

For rollout buffers, rather than skipping, store `autoreset` alongside each step and use it to mask the loss and to stop advantage or return calculations crossing an episode boundary.

(autoreset-same-step)=
## Same-step mode

On the step that a sub-environment's episode ends, the sub-environment is reset within the same `step()` call. The returned observation and info are from the reset, while the reward and `terminated`/`truncated` are from the episode's final step. The final observations are given in `info["final_obs"]`, a NumPy object array indexed by sub-environment, and the final infos in `info["final_info"]`, batched in the same format as the step's info. The boolean masks `info["_final_obs"]` and `info["_final_info"]` are `True` for the sub-environments whose episode ended. These keys are only in `info` on steps where an episode ends.

Every step is a real environment transition, so no steps need to be skipped, but the next observation of a finished sub-environment must be taken from `info["final_obs"]`. This matters for truncated episodes, where the value of the final observation is bootstrapped.

```python
import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

envs = gym.make_vec(
    "CartPole-v1",
    num_envs=4,
    vectorization_mode="sync",
    vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
)
replay_buffer = []

obs, infos = envs.reset(seed=42)
for _ in range(1_000):
    actions = envs.action_space.sample()  # replace with your policy, `policy(obs)`
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    # `next_obs` contains reset observations for the sub-environments that ended, replace them with the final observations
    final_next_obs = next_obs.copy()
    if "final_obs" in infos:
        for j in np.flatnonzero(infos["_final_obs"]):
            final_next_obs[j] = infos["final_obs"][j]

    for j in range(envs.num_envs):
        replay_buffer.append((obs[j], actions[j], rewards[j], terminations[j], final_next_obs[j]))

    obs = next_obs  # the reset observations start the next episodes
envs.close()
```

```{note}
Some vector wrappers don't support same-step autoreset as they don't transform `info["final_obs"]`, see {ref}`vector-wrappers-autoreset-support`.
```

(autoreset-disabled)=
## Disabled mode

No sub-environment is reset automatically. On the step that a sub-environment's episode ends, `step()` returns its final observation, reward, `terminated`/`truncated` and info, and the user must reset it before stepping it again by passing a boolean mask of shape `(num_envs,)` to `reset(options={"reset_mask": mask})`. Only the masked sub-environments are reset. The returned observations contain the reset observations for masked sub-environments and the latest observations for the others, while the returned info only contains data for the reset sub-environments. Stepping a sub-environment that has ended without resetting it is an error, `SyncVectorEnv` raises an `AssertionError` while `AsyncVectorEnv` doesn't check.

As in single environment training, every step is a real transition and the next observation is the final observation, at the cost of an additional `reset` call when episodes end.

```python
import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

envs = gym.make_vec(
    "CartPole-v1",
    num_envs=4,
    vectorization_mode="sync",
    vector_kwargs={"autoreset_mode": AutoresetMode.DISABLED},
)
replay_buffer = []

obs, infos = envs.reset(seed=42)
for _ in range(1_000):
    actions = envs.action_space.sample()  # replace with your policy, `policy(obs)`
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    for j in range(envs.num_envs):
        replay_buffer.append((obs[j], actions[j], rewards[j], terminations[j], next_obs[j]))

    # reset the sub-environments whose episodes ended
    reset_mask = np.logical_or(terminations, truncations)
    if np.any(reset_mask):
        next_obs, reset_infos = envs.reset(options={"reset_mask": reset_mask})
    obs = next_obs
envs.close()
```

## Summary

For a sub-environment whose episode ends on step `t`:

| | Next-step (default) | Same-step | Disabled |
|---|---|---|---|
| Observation at step `t` | Final observation | Reset observation | Final observation |
| Final observation | Step `t` observation | `info["final_obs"][i]` (masked by `info["_final_obs"]`) | Step `t` observation |
| Final info | Step `t` info | `info["final_info"]` (masked by `info["_final_info"]`) | Step `t` info |
| Reward, `terminated`, `truncated` at step `t` | From the final step | From the final step | From the final step |
| Reset | On step `t + 1`, its action is ignored, returning reward `0` and `terminated = truncated = False` | Within step `t` | By the user, `reset(options={"reset_mask": mask})` |
| Training code | Skip or mask step `t + 1` using step `t`'s `terminated or truncated` | Use `info["final_obs"]` as the next observation | Reset sub-environments with `reset_mask` |

The single-environment {py:class}`gymnasium.wrappers.Autoreset` wrapper supports the next-step and same-step modes with the same behaviour, where same-step adds `info["final_obs"]` and `info["final_info"]` without masks.

For further background on the design of the autoreset modes, see the Farama blog post, [Deep Dive: Autoreset Modes in Gymnasium v1.1 Vector Environments](https://farama.org/Vector-Autoreset-Mode).

## API

```{eval-rst}
.. autoclass:: gymnasium.vector.AutoresetMode
    :members:
```
