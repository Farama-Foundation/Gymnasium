"""A collection of runtime performance bencharks, useful for debugging performance related issues."""

import time
from collections.abc import Callable

import numpy as np

import gymnasium
from gymnasium.vector import AutoresetMode, VectorEnv


def benchmark_step(
    env: gymnasium.Env, target_duration: int = 5, seed: int | None = None
) -> float:
    """A benchmark to measure the runtime performance of step for an environment.

    example usage:
        ```py
        env_old = ...
        old_throughput = benchmark_step(env_old)
        env_new = ...
        new_throughput = benchmark_step(env_old)
        slowdown = old_throughput / new_throughput
        ```

    Args:
        env: the environment to benchmarked.
        target_duration: the duration of the benchmark in seconds (note: it will go slightly over it).
        seed: seeds the environment and action sampled.

    Returns: the average steps per second.
    """
    steps = 0
    end = 0.0
    env.reset(seed=seed)
    env.action_space.sample()
    start = time.time()

    while True:
        steps += 1
        action = env.action_space.sample()
        _, _, terminal, truncated, _ = env.step(action)

        if terminal or truncated:
            env.reset()

        if time.time() - start > target_duration:
            end = time.time()
            break

    length = end - start

    steps_per_time = steps / length
    return steps_per_time


def benchmark_step_vector(
    env: VectorEnv, target_duration: int = 5, seed: int | None = None
) -> float:
    """Measure the step throughput of a vector environment.

    Args:
        env: The vector environment to benchmark.
        target_duration: The benchmark duration in seconds. The benchmark can run
            slightly longer while its final step completes.
        seed: Seed for the environment and action space.

    Returns:
        The number of individual environment steps per second.
    """
    env.action_space.seed(seed)
    env.reset(seed=seed)

    # Warm up lazy initialization, including JIT compilation, outside the benchmark.
    env.step(env.action_space.sample())
    env.reset(seed=seed)

    steps = 0
    start = time.time()
    autoreset_mode = env.metadata.get("autoreset_mode", AutoresetMode.NEXT_STEP)

    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += env.num_envs

        if autoreset_mode == AutoresetMode.DISABLED:
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                env.reset(options={"reset_mask": done})

        if time.time() - start > target_duration:
            end = time.time()
            break

    return steps / (end - start)


def benchmark_init(
    env_lambda: Callable[[], gymnasium.Env],
    target_duration: int = 5,
    seed: int | None = None,
) -> float:
    """A benchmark to measure the initialization time and first reset.

    Args:
        env_lambda: the function to initialize the environment.
        target_duration: the duration of the benchmark in seconds (note: it will go slightly over it).
        seed: seeds the first reset of the environment.
    """
    inits = 0
    end = 0.0
    start = time.time()
    while True:
        inits += 1
        env = env_lambda()
        env.reset(seed=seed)

        if time.time() - start > target_duration:
            end = time.time()
            break
    length = end - start

    inits_per_time = inits / length
    return inits_per_time


def benchmark_render(env: gymnasium.Env, target_duration: int = 5) -> float:
    """A benchmark to measure the time of render().

    Note: does not work with `render_mode='human'`
    Args:
        env: the environment to benchmarked (Note: must be renderable).
        target_duration: the duration of the benchmark in seconds (note: it will go slightly over it).

    """
    renders = 0
    end = 0.0
    start = time.time()
    while True:
        renders += 1
        env.render()

        if time.time() - start > target_duration:
            end = time.time()
            break
    length = end - start

    renders_per_time = renders / length
    return renders_per_time
