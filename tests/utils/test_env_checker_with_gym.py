import re

import pytest

import gymnasium
from gymnasium.utils.env_checker import check_env


gym = pytest.importorskip("gym")


class NoClassEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)


class IncorrectEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)


class IncorrectAction(gymnasium.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gymnasium.spaces.Discrete(2)


class IncorrectObs(gymnasium.Env):
    def __init__(self):
        self.action_space = gymnasium.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)


def test_check_env_with_gym():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "The environment must inherit from the gymnasium.Env class, actual class: <class"
        ),
    ):
        check_env(NoClassEnv())

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Gym is incompatible with Gymnasium, please update the environment class to `gymnasium.Env`."
        ),
    ):
        check_env(IncorrectEnv())

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Gym is incompatible with Gymnasium, please update the environment observation_space to `<class 'gymnasium.spaces.space.Space'>`."
        ),
    ):
        check_env(IncorrectObs())

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Gym is incompatible with Gymnasium, please update the environment action_space to `<class 'gymnasium.spaces.space.Space'>`."
        ),
    ):
        check_env(IncorrectAction())


def test_passive_env_checker_with_gym():
    gymnasium.register("NoClassEnv", NoClassEnv)
    gymnasium.register("IncorrectEnv", IncorrectEnv)
    gymnasium.register("IncorrectObs", IncorrectObs)
    gymnasium.register("IncorrectAction", IncorrectAction)

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The environment must inherit from the gymnasium.Env class, actual class: <class"
        ),
    ):
        gymnasium.make("NoClassEnv")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Gym is incompatible with Gymnasium, please update the environment class to `gymnasium.Env`."
        ),
    ):
        gymnasium.make("IncorrectEnv")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Gym is incompatible with Gymnasium, please update the environment observation_space to `<class 'gymnasium.spaces.space.Space'>`."
        ),
    ):
        gymnasium.make("IncorrectObs")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Gym is incompatible with Gymnasium, please update the environment action_space to `<class 'gymnasium.spaces.space.Space'>`."
        ),
    ):
        gymnasium.make("IncorrectAction")

    gymnasium.registry.pop("NoClassEnv")
    gymnasium.registry.pop("IncorrectEnv")
    gymnasium.registry.pop("IncorrectObs")
    gymnasium.registry.pop("IncorrectAction")
