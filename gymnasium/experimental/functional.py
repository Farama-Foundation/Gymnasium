"""Base class and definitions for an alternative, functional backend for gym envs, particularly suitable for hardware accelerated and otherwise transformed environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

import numpy as np

from gymnasium import Space


StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")
RewardType = TypeVar("RewardType")
TerminalType = TypeVar("TerminalType")
RenderStateType = TypeVar("RenderStateType")
Params = TypeVar("Params")


class FuncEnv(
    Generic[
        StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType, Params
    ]
):
    """Base class (template) for functional envs.

    This API is meant to be used in a stateless manner, with the environment state being passed around explicitly.
    That being said, nothing here prevents users from using the environment statefully, it's just not recommended.
    A functional env consists of the following functions (in this case, instance methods):

     * initial: returns the initial state of the POMDP
     * observation: returns the observation in a given state
     * transition: returns the next state after taking an action in a given state
     * reward: returns the reward for a given (state, action, next_state) tuple
     * terminal: returns whether a given state is terminal
     * state_info: optional, returns a dict of info about a given state
     * step_info: optional, returns a dict of info about a given (state, action, next_state) tuple

    The class-based structure serves the purpose of allowing environment constants to be defined in the class,
    and then using them by name in the code itself.

    For the moment, this is predominantly for internal use. This API is likely to change, but in the future
    we intend to flesh it out and officially expose it to end users.
    """

    observation_space: Space
    action_space: Space

    def __init__(self, options: dict[str, Any] | None = None):
        """Initialize the environment constants."""
        self.__dict__.update(options or {})
        self.default_params = self.get_default_params()

    def initial(self, rng: Any, params: Params | None = None) -> StateType:
        """Generates the initial state of the environment with a random number generator."""
        raise NotImplementedError

    def transition(
        self, state: StateType, action: ActType, rng: Any, params: Params | None = None
    ) -> StateType:
        """Updates (transitions) the state with an action and random number generator."""
        raise NotImplementedError

    def observation(
        self, state: StateType, rng: Any, params: Params | None = None
    ) -> ObsType:
        """Generates an observation for a given state of an environment."""
        raise NotImplementedError

    def reward(
        self,
        state: StateType,
        action: ActType,
        next_state: StateType,
        rng: Any,
        params: Params | None = None,
    ) -> RewardType:
        """Computes the reward for a given transition between `state`, `action` to `next_state`."""
        raise NotImplementedError

    def terminal(
        self, state: StateType, rng: Any, params: Params | None = None
    ) -> TerminalType:
        """Returns if the state is a final terminal state."""
        raise NotImplementedError

    def state_info(self, state: StateType, params: Params | None = None) -> dict:
        """Info dict about a single state."""
        return {}

    def transition_info(
        self,
        state: StateType,
        action: ActType,
        next_state: StateType,
        params: Params | None = None,
    ) -> dict:
        """Info dict about a full transition."""
        return {}

    def transform(self, func: Callable[[Callable], Callable]):
        """Functional transformations."""
        self.initial = func(self.initial)
        self.transition = func(self.transition)
        self.observation = func(self.observation)
        self.reward = func(self.reward)
        self.terminal = func(self.terminal)
        self.state_info = func(self.state_info)
        self.step_info = func(self.transition_info)

    def render_image(
        self,
        state: StateType,
        render_state: RenderStateType,
        params: Params | None = None,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Show the state."""
        raise NotImplementedError

    def render_init(self, params: Params | None = None, **kwargs) -> RenderStateType:
        """Initialize the render state."""
        raise NotImplementedError

    def render_close(self, render_state: RenderStateType, params: Params | None = None):
        """Close the render state."""
        raise NotImplementedError

    def get_default_params(self, **kwargs) -> Params | None:
        """Get the default params."""
        return None
