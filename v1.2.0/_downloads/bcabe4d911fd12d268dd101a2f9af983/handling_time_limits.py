"""
Handling Time Limits
====================

This tutorial explains how time limits should be correctly handled with `termination` and `truncation` signals.

The ``done`` signal received (in previous versions of OpenAI Gym < 0.26) from ``env.step`` indicated whether an episode has ended.
However, this signal did not distinguish whether the episode ended due to ``termination`` or ``truncation``.

Termination
-----------

Termination refers to the episode ending after reaching a terminal state that is defined as part of the environment
definition. Examples are - task success, task failure, robot falling down etc. Notably, this also includes episodes
ending in finite-horizon environments due to a time-limit inherent to the environment. Note that to preserve Markov
property, a representation of the remaining time must be present in the agent's observation in finite-horizon environments.
`(Reference) <https://arxiv.org/abs/1712.00378>`_

Truncation
----------

Truncation refers to the episode ending after an externally defined condition (that is outside the scope of the Markov
Decision Process). This could be a time-limit, a robot going out of bounds etc.

An infinite-horizon environment is an obvious example of where this is needed. We cannot wait forever for the episode
to complete, so we set a practical time-limit after which we forcibly halt the episode. The last state in this case is
not a terminal state since it has a non-zero transition probability of moving to another state as per the Markov
Decision Process that defines the RL problem. This is also different from time-limits in finite horizon environments
as the agent in this case has no idea about this time-limit.
"""

# %%
# Importance in learning code
# ---------------------------
# Bootstrapping (using one or more estimated values of a variable to update estimates of the same variable) is a key
# aspect of Reinforcement Learning. A value function will tell you how much discounted reward you will get from a
# particular state if you follow a given policy. When an episode stops at any given point, by looking at the value of
# the final state, the agent is able to estimate how much discounted reward could have been obtained if the episode has
# continued. This is an example of handling truncation.
#
# More formally, a common example of bootstrapping in RL is updating the estimate of the Q-value function,
#
# .. math::
#   Q_{target}(o_t, a_t) = r_t + \gamma . \max_a(Q(o_{t+1}, a_{t+1}))
#
#
# In classical RL, the new ``Q`` estimate is a weighted average of the previous ``Q`` estimate and ``Q_target`` while in Deep
# Q-Learning, the error between ``Q_target`` and the previous ``Q`` estimate is minimized.
#
# However, at the terminal state, bootstrapping is not done,
#
# .. math::
#   Q_{target}(o_t, a_t) = r_t
#
# This is where the distinction between termination and truncation becomes important. When an episode ends due to
# termination we don't bootstrap, when it ends due to truncation, we bootstrap.
#
# While using gymnasium environments, the ``done`` signal (default for < v0.26) is frequently used to determine whether to
# bootstrap or not. However, this is incorrect since it does not differentiate between termination and truncation.
#
# A simple example of value functions is shown below. This is an illustrative example and not part of any specific algorithm.
#
# .. code:: python
#
#   # INCORRECT
#   vf_target = rew + gamma * (1 - done) * vf_next_state
#
# This is incorrect in the case of episode ending due to a truncation, where bootstrapping needs to happen but it doesn't.

# %%
# Solution
# ----------
#
# From v0.26 onwards, Gymnasium's ``env.step`` API returns both termination and truncation information explicitly.
# In the previous version truncation information was supplied through the info key ``TimeLimit.truncated``.
# The correct way to handle terminations and truncations now is,
#
# .. code:: python
#
#   # terminated = done and 'TimeLimit.truncated' not in info
#   # This was needed in previous versions.
#
#   vf_target = rew + gamma * (1 - terminated) * vf_next_state
