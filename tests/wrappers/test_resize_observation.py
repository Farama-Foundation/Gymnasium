from __future__ import annotations

import pytest

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import GrayscaleObservationV0, ResizeObservationV0


@pytest.mark.parametrize("env_id", ["CarRacing-v2"])
@pytest.mark.parametrize("shape", [(8, 5), (10, 7)])
def test_resize_observation(env_id, shape: tuple[int, int]):
    base_env = gym.make(env_id, disable_env_checker=True)
    env = ResizeObservationV0(base_env, shape)

    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape[-1] == 3
    obs, _ = env.reset()

    assert env.observation_space.shape[:2] == tuple(shape)
    assert obs.shape == tuple(shape) + (3,)

    # test two-dimensional input by grayscaling the observation
    gray_env = GrayscaleObservationV0(base_env, keep_dim=False)
    env = ResizeObservationV0(gray_env, shape)
    obs, _ = env.reset()
    if isinstance(shape, int):
        assert env.observation_space.shape == obs.shape == (shape, shape)
    else:
        assert env.observation_space.shape == obs.shape == tuple(shape)


def test_invalid_input():
    env = gym.make("CarRacing-v2", disable_env_checker=True)
    with pytest.raises(AssertionError):
        ResizeObservationV0(env, ())
    with pytest.raises(AssertionError):
        ResizeObservationV0(env, (1,))
    with pytest.raises(AssertionError):
        ResizeObservationV0(env, (1, 1, 1, 1))
    with pytest.raises(AssertionError):
        ResizeObservationV0(env, (-1, 1))
    with pytest.raises(AssertionError):
        ResizeObservationV0(gym.make("CartPole-v1", disable_env_checker=True), (1, 1))
    with pytest.raises(AssertionError):
        ResizeObservationV0(gym.make("Blackjack-v1", disable_env_checker=True), (1, 1))
