"""Module for vector environments."""
from typing import Callable, Iterable, List, Optional, Union

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.vector_env import VectorEnv, VectorEnvWrapper


__all__ = ["AsyncVectorEnv", "SyncVectorEnv", "VectorEnv", "VectorEnvWrapper", "make"]


def make(
    id: str,
    num_envs: int = 1,
    asynchronous: bool = True,
    wrappers: Optional[Union[Callable[[Env], Env], List[Callable[[Env], Env]]]] = None,
    disable_env_checker: Optional[bool] = None,
    **kwargs,
) -> VectorEnv:
    """Create a vectorized environment from multiple copies of an environment, from its id.

    Example::

        >>> import gymnasium as gym
        >>> env = gym.vector.make('CartPole-v1', num_envs=3)
        >>> env.reset(seed=42)
        (array([[ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ],
               [ 0.01522993, -0.04562247, -0.04799704,  0.03392126],
               [-0.03774345, -0.02418869, -0.00942293,  0.0469184 ]],
              dtype=float32), {})

    Args:
        id: The environment ID. This must be a valid ID from the registry.
        num_envs: Number of copies of the environment.
        asynchronous: If `True`, wraps the environments in an :class:`AsyncVectorEnv` (which uses `multiprocessing` to run the environments in parallel). If ``False``, wraps the environments in a :class:`SyncVectorEnv`.
        wrappers: If not ``None``, then apply the wrappers to each internal environment during creation.
        disable_env_checker: If to run the env checker for the first environment only. None will default to the environment spec `disable_env_checker` parameter
            (that is by default False), otherwise will run according to this argument (True = not run, False = run)
        **kwargs: Keywords arguments applied during `gym.make`

    Returns:
        The vectorized environment.
    """

    def create_env(env_num: int) -> Callable[[], Env]:
        """Creates an environment that can enable or disable the environment checker."""
        # If the env_num > 0 then disable the environment checker otherwise use the parameter
        _disable_env_checker = True if env_num > 0 else disable_env_checker

        def _make_env() -> Env:
            env = gym.envs.registration.make(
                id,
                disable_env_checker=_disable_env_checker,
                **kwargs,
            )
            if wrappers is not None:
                if callable(wrappers):
                    env = wrappers(env)
                elif isinstance(wrappers, Iterable) and all(
                    [callable(w) for w in wrappers]
                ):
                    for wrapper in wrappers:
                        env = wrapper(env)
                else:
                    raise NotImplementedError
            return env

        return _make_env

    env_fns = [
        create_env(disable_env_checker or env_num > 0) for env_num in range(num_envs)
    ]
    return AsyncVectorEnv(env_fns) if asynchronous else SyncVectorEnv(env_fns)
