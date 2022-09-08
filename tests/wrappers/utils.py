import gymnasium


def has_wrapper(wrapped_env: gymnasium.Env, wrapper_type: type) -> bool:
    while isinstance(wrapped_env, gymnasium.Wrapper):
        if isinstance(wrapped_env, wrapper_type):
            return True
        wrapped_env = wrapped_env.env
    return False
