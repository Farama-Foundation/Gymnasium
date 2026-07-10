"""Module for 2d physics environments with functional and environment implementations."""

from gymnasium.error import DependencyNotInstalled

try:
    from gymnasium.envs.phys2d.cartpole import CartPoleFunctional, CartPoleJaxEnv
    from gymnasium.envs.phys2d.pendulum import PendulumFunctional, PendulumJaxEnv
except ImportError as e:
    raise DependencyNotInstalled(
        "Phys2d environments require Jax. Install it with `pip install jax`."
    ) from e

__all__ = [
    "CartPoleFunctional",
    "CartPoleJaxEnv",
    "PendulumFunctional",
    "PendulumJaxEnv",
]
