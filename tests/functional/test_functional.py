"""Tests the functional api."""

from __future__ import annotations

from typing import Any

import numpy as np

from gymnasium.experimental.functional import FuncEnv


class GenericTestFuncEnv(FuncEnv):
    """Generic testing functional environment."""

    def __init__(self, options: dict[str, Any] | None = None):
        """Constructor that allows generic options to be set on the environment."""
        super().__init__(options)

    def initial(self, rng: Any, params=None) -> np.ndarray:
        """Testing initial function."""
        return np.array([0, 0], dtype=np.float32)

    def observation(self, state: np.ndarray, rng: Any, params=None) -> np.ndarray:
        """Testing observation function."""
        return state

    def transition(
        self, state: np.ndarray, action: int, rng: None, params=None
    ) -> np.ndarray:
        """Testing transition function."""
        return state + np.array([0, action], dtype=np.float32)

    def reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        rng: Any,
        params=None,
    ) -> float:
        """Testing reward function."""
        return 1.0 if next_state[1] > 0 else 0.0

    def terminal(self, state: np.ndarray, rng: Any, params=None) -> bool:
        """Testing terminal function."""
        return state[1] > 0


def test_functional_api():
    """Tests the core functional api specification using a generic testing environment."""
    env = GenericTestFuncEnv()

    state = env.initial(None)

    obs = env.observation(state, None)

    assert state.shape == (2,)
    assert state.dtype == np.float32
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert np.allclose(obs, state)

    actions = [-1, -2, -5, 3, 5, 2]
    for i, action in enumerate(actions):
        next_state = env.transition(state, action, None)
        assert next_state.shape == (2,)
        assert next_state.dtype == np.float32
        assert np.allclose(next_state, state + np.array([0, action]))

        observation = env.observation(next_state, None)
        assert observation.shape == (2,)
        assert observation.dtype == np.float32
        assert np.allclose(observation, next_state)

        reward = env.reward(state, action, next_state, None)
        assert reward == (1.0 if next_state[1] > 0 else 0.0)

        terminal = env.terminal(next_state, None)
        assert terminal == (i == 5)  # terminal state is in the final action

        state = next_state
