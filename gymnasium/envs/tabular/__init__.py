"""Provides Tabular JAX FuncEnv implementations."""

from gymnasium.error import DependencyNotInstalled

try:
    from gymnasium.envs.tabular.blackjack import BlackJackJaxEnv
    from gymnasium.envs.tabular.cliffwalking import CliffWalkingJaxEnv
except ImportError as e:
    raise DependencyNotInstalled(
        "Tabular environments require Jax. Install it with `pip install jax`."
    ) from e

__all__ = [
    "BlackJackJaxEnv",
    "CliffWalkingJaxEnv",
]
