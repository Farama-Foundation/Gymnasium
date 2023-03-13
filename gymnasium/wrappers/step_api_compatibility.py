"""Implementation of StepAPICompatibility wrapper class for transforming envs between new and old step API."""
import gymnasium as gym
from gymnasium.logger import deprecation
from gymnasium.utils.step_api_compatibility import step_api_compatibility


class StepAPICompatibility(gym.Wrapper, gym.utils.RecordConstructorArgs):
    r"""A wrapper which can transform an environment from new step API to old and vice-versa.

    Old step API refers to step() method returning (observation, reward, done, info)
    New step API refers to step() method returning (observation, reward, terminated, truncated, info)
    (Refer to docs for details on the API change)

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import StepAPICompatibility
        >>> env = gym.make("CartPole-v1")
        >>> env # wrapper not applied by default, set to new API
        <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
        >>> env = StepAPICompatibility(gym.make("CartPole-v1"))
        >>> env
        <StepAPICompatibility<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>>
    """

    def __init__(self, env: gym.Env, output_truncation_bool: bool = True):
        """A wrapper which can transform an environment from new step API to old and vice-versa.

        Args:
            env (gym.Env): the env to wrap. Can be in old or new API
            output_truncation_bool (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, output_truncation_bool=output_truncation_bool
        )
        gym.Wrapper.__init__(self, env)

        self.is_vector_env = isinstance(env.unwrapped, gym.vector.VectorEnv)
        self.output_truncation_bool = output_truncation_bool
        if not self.output_truncation_bool:
            deprecation(
                "Initializing environment in (old) done step API which returns one bool instead of two."
            )

    def step(self, action):
        """Steps through the environment, returning 5 or 4 items depending on `output_truncation_bool`.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info) or (observation, reward, done, info)
        """
        step_returns = self.env.step(action)
        return step_api_compatibility(
            step_returns, self.output_truncation_bool, self.is_vector_env
        )
