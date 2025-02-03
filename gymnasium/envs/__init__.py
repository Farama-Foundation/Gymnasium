"""Registers the internal gym envs then loads the env plugins for module using the entry point."""

from typing import Any

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec



# Toy Text
# ----------------------------------------

register(
    id="Blackjack-v1",
    entry_point="gymnasium.envs.toy_text.blackjack:BlackjackEnv",
    kwargs={"sab": True, "natural": False},
)

# Tabular
# ----------------------------------------

register(
    id="tabular/Blackjack-v0",
    entry_point="gymnasium.envs.tabular.blackjack:BlackJackJaxEnv",
    disable_env_checker=True,
)

# --- For shimmy compatibility
def _raise_shimmy_error(*args: Any, **kwargs: Any):
    raise ImportError(
        'To use the gym compatibility environments, run `pip install "shimmy[gym-v21]"` or `pip install "shimmy[gym-v26]"`'
    )


# When installed, shimmy will re-register these environments with the correct entry_point
register(id="GymV21Environment-v0", entry_point=_raise_shimmy_error)
register(id="GymV26Environment-v0", entry_point=_raise_shimmy_error)
