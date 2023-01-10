import pytest

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation


@pytest.mark.parametrize("env_id", ["CarRacing-v2"])
@pytest.mark.parametrize("shape", [16, 32, (8, 5), [10, 7]])
def test_resize_observation(env_id, shape):
    base_env = gym.make(env_id, disable_env_checker=True)
    env = ResizeObservation(base_env, shape)

    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape[-1] == 3
    obs, _ = env.reset()
    if isinstance(shape, int):
        assert env.observation_space.shape[:2] == (shape, shape)
        assert obs.shape == (shape, shape, 3)
    else:
        assert env.observation_space.shape[:2] == tuple(shape)
        assert obs.shape == tuple(shape) + (3,)

    # test two-dimensional input by grayscaling the observation
    gray_env = GrayScaleObservation(base_env, keep_dim=False)
    env = ResizeObservation(gray_env, shape)
    obs, _ = env.reset()
    if isinstance(shape, int):
        assert env.observation_space.shape == obs.shape == (shape, shape)
    else:
        assert env.observation_space.shape == obs.shape == tuple(shape)


def test_invalid_input():
    env = gym.make("CarRacing-v2", disable_env_checker=True)
    with pytest.raises(AssertionError):
        ResizeObservation(env, ())
    with pytest.raises(AssertionError):
        ResizeObservation(env, (1,))
    with pytest.raises(AssertionError):
        ResizeObservation(env, (1, 1, 1, 1))
    with pytest.raises(AssertionError):
        ResizeObservation(env, -1)
    with pytest.raises(AssertionError):
        ResizeObservation(gym.make("CartPole-v1", disable_env_checker=True), 1)
    with pytest.raises(AssertionError):
        ResizeObservation(gym.make("Blackjack-v1", disable_env_checker=True), 1)
