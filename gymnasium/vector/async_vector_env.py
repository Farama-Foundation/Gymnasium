"""An async vector environment."""

from __future__ import annotations

import multiprocessing
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from copy import deepcopy
from enum import Enum
from multiprocessing import Queue
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import SynchronizedArray
from typing import Any

import numpy as np

from gymnasium import Space, logger
from gymnasium.core import ActType, Env, ObsType, RenderFrame
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
from gymnasium.vector.utils import (
    CloudpickleWrapper,
    batch_differing_spaces,
    batch_space,
    clear_mpi_env_vars,
    concatenate,
    create_empty_array,
    create_shared_memory,
    iterate,
    read_from_shared_memory,
    write_to_shared_memory,
)
from gymnasium.vector.vector_env import ArrayType, AutoresetMode, VectorEnv


__all__ = ["AsyncVectorEnv", "AsyncState"]


class AsyncState(Enum):
    """The AsyncVectorEnv possible states given the different actions."""

    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel.

    It uses ``multiprocessing`` processes, and pipes for communication.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("Pendulum-v1", num_envs=2, vectorization_mode="async")
        >>> envs
        AsyncVectorEnv(Pendulum-v1, num_envs=2)
        >>> envs = gym.vector.AsyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> envs
        AsyncVectorEnv(num_envs=2)
        >>> observations, infos = envs.reset(seed=42)
        >>> observations
        array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32)
        >>> infos
        {}
        >>> _ = envs.action_space.seed(123)
        >>> observations, rewards, terminations, truncations, infos = envs.step(envs.action_space.sample())
        >>> observations
        array([[-0.1851753 ,  0.98270553,  0.714599  ],
               [ 0.6193494 ,  0.7851154 , -1.0808398 ]], dtype=float32)
        >>> rewards
        array([-2.96495728, -1.00214607])
        >>> terminations
        array([False, False])
        >>> truncations
        array([False, False])
        >>> infos
        {}
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        shared_memory: bool = True,
        copy: bool = True,
        context: str | None = None,
        daemon: bool = True,
        worker: (
            Callable[
                [int, Callable[[], Env], Connection, Connection, bool, Queue], None
            ]
            | None
        ) = None,
        observation_mode: str | Space = "same",
        autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP,
    ):
        """Vectorized environment that runs multiple environments in parallel.

        Args:
            env_fns: Functions that create the environments.
            shared_memory: If ``True``, then the observations from the worker processes are communicated back through
                shared variables. This can improve the efficiency if the observations are large (e.g. images).
            copy: If ``True``, then the :meth:`AsyncVectorEnv.reset` and :meth:`AsyncVectorEnv.step` methods
                return a copy of the observations.
            context: Context for `multiprocessing`. If ``None``, then the default context is used.
            daemon: If ``True``, then subprocesses have ``daemon`` flag turned on; that is, they will quit if
                the head process quits. However, ``daemon=True`` prevents subprocesses to spawn children,
                so for some environments you may want to have it set to ``False``.
            worker: If set, then use that worker in a subprocess instead of a default one.
                Can be useful to override some inner vector env logic, for instance, how resets on termination or truncation are handled.
            observation_mode: Defines how environment observation spaces should be batched. 'same' defines that there should be ``n`` copies of identical spaces.
                'different' defines that there can be multiple observation spaces with different parameters though requires the same shape and dtype,
                warning, may raise unexpected errors. Passing a ``Tuple[Space, Space]`` object allows defining a custom ``single_observation_space`` and
                ``observation_space``, warning, may raise unexpected errors.
            autoreset_mode: The Autoreset Mode used, see https://farama.org/Vector-Autoreset-Mode for more information.

        Warnings:
            worker is an advanced mode option. It provides a high degree of flexibility and a high chance
            to shoot yourself in the foot; thus, if you are writing your own worker, it is recommended to start
            from the code for ``_worker`` (or ``_async_worker``) method, and add changes.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
            ValueError: If observation_space is a custom space (i.e. not a default space in Gym,
                such as gymnasium.spaces.Box, gymnasium.spaces.Discrete, or gymnasium.spaces.Dict) and shared_memory is True.
        """
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        self.context = context
        self.daemon = daemon
        self.worker = worker
        self.observation_mode = observation_mode
        self.autoreset_mode = (
            autoreset_mode
            if isinstance(autoreset_mode, AutoresetMode)
            else AutoresetMode(autoreset_mode)
        )

        self.num_envs = len(env_fns)

        # This would be nice to get rid of, but without it there's a deadlock between shared memory and pipes
        # Create a dummy environment to gather the metadata and observation / action space of the environment
        dummy_env = env_fns[0]()

        # As we support `make_vec(spec)` then we can't include a `spec = dummy_env.spec` as this doesn't guarantee we can actual recreate the vector env.
        self.metadata = dummy_env.metadata
        self.metadata["autoreset_mode"] = self.autoreset_mode
        self.render_mode = dummy_env.render_mode

        self.single_action_space = dummy_env.action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        if isinstance(observation_mode, tuple) and len(observation_mode) == 2:
            assert isinstance(observation_mode[0], Space)
            assert isinstance(observation_mode[1], Space)
            self.observation_space, self.single_observation_space = observation_mode
        else:
            if observation_mode == "same":
                self.single_observation_space = dummy_env.observation_space
                self.observation_space = batch_space(
                    self.single_observation_space, self.num_envs
                )
            elif observation_mode == "different":
                # the environment is created and instantly destroy, might cause issues for some environment
                # but I don't believe there is anything else we can do, for users with issues, pre-compute the spaces and use the custom option.
                env_spaces = [env().observation_space for env in self.env_fns]

                self.single_observation_space = env_spaces[0]
                self.observation_space = batch_differing_spaces(env_spaces)
            else:
                raise ValueError(
                    f"Invalid `observation_mode`, expected: 'same' or 'different' or tuple of single and batch observation space, actual got {observation_mode}"
                )

        dummy_env.close()
        del dummy_env

        # Generate the multiprocessing context for the observation buffer
        ctx = multiprocessing.get_context(context)
        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    self.single_observation_space, _obs_buffer, n=self.num_envs
                )
            except CustomSpaceError as e:
                raise ValueError(
                    "Using `AsyncVector(..., shared_memory=True)` caused an error, you can disable this feature with `shared_memory=False` however this is slower."
                ) from e
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = worker or _async_worker
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                        self.autoreset_mode,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_spaces()

    @property
    def np_random_seed(self) -> tuple[int, ...]:
        """Returns a tuple of np_random seeds for all the wrapped envs."""
        return self.get_attr("np_random_seed")

    @property
    def np_random(self) -> tuple[np.random.Generator, ...]:
        """Returns the tuple of the numpy random number generators for the wrapped envs."""
        return self.get_attr("np_random")

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets all sub-environments in parallel and return a batch of concatenated observations and info.

        Args:
            seed: The environment reset seeds
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        self.reset_async(seed=seed, options=options)
        return self.reset_wait()

    def reset_async(
        self,
        seed: int | list[int | None] | None = None,
        options: dict | None = None,
    ):
        """Send calls to the :obj:`reset` methods of the sub-environments.

        To get the results of these calls, you may invoke :meth:`reset_wait`.

        Args:
            seed: List of seeds for each environment
            options: The reset option

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: If the environment is already waiting for a pending call to another
                method (e.g. :meth:`step_async`). This can be caused by two consecutive
                calls to :meth:`reset_async`, with no call to :meth:`reset_wait` in between.
        """
        self._assert_is_running()

        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert (
            len(seed) == self.num_envs
        ), f"If seeds are passed as a list the length must match num_envs={self.num_envs} but got length={len(seed)}."

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
                str(self._state.value),
            )

        if options is not None and "reset_mask" in options:
            reset_mask = options.pop("reset_mask")
            assert isinstance(
                reset_mask, np.ndarray
            ), f"`options['reset_mask': mask]` must be a numpy array, got {type(reset_mask)}"
            assert reset_mask.shape == (
                self.num_envs,
            ), f"`options['reset_mask': mask]` must have shape `({self.num_envs},)`, got {reset_mask.shape}"
            assert (
                reset_mask.dtype == np.bool_
            ), f"`options['reset_mask': mask]` must have `dtype=np.bool_`, got {reset_mask.dtype}"
            assert np.any(
                reset_mask
            ), f"`options['reset_mask': mask]` must contain a boolean array, got reset_mask={reset_mask}"

            for pipe, env_seed, env_reset in zip(self.parent_pipes, seed, reset_mask):
                if env_reset:
                    env_kwargs = {"seed": env_seed, "options": options}
                    pipe.send(("reset", env_kwargs))
                else:
                    pipe.send(("reset-noop", None))
        else:
            for pipe, env_seed in zip(self.parent_pipes, seed):
                env_kwargs = {"seed": env_seed, "options": options}
                pipe.send(("reset", env_kwargs))

        self._state = AsyncState.WAITING_RESET

    def reset_wait(
        self,
        timeout: int | float | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            timeout: Number of seconds before the call to ``reset_wait`` times out. If `None`, the call to ``reset_wait`` never times out.

        Returns:
            A tuple of batched observations and list of dictionaries

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`reset_wait` was called without any prior call to :meth:`reset_async`.
            TimeoutError: If :meth:`reset_wait` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(
                f"The call to `reset_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

        infos = {}
        results, info_data = zip(*results)
        for i, info in enumerate(info_data):
            infos = self._add_info(infos, info, i)

        if not self.shared_memory:
            self.observations = concatenate(
                self.single_observation_space, results, self.observations
            )

        self._state = AsyncState.DEFAULT
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Take an action for each parallel environment.

        Args:
            actions: element of :attr:`action_space` batch of actions.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: np.ndarray):
        """Send the calls to :meth:`Env.step` to each sub-environment.

        Args:
            actions: Batch of actions. element of :attr:`VectorEnv.action_space`

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: If the environment is already waiting for a pending call to another
                method (e.g. :meth:`reset_async`). This can be caused by two consecutive
                calls to :meth:`step_async`, with no call to :meth:`step_wait` in
                between.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )

        iter_actions = iterate(self.action_space, actions)
        for pipe, action in zip(self.parent_pipes, iter_actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(
        self, timeout: int | float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Wait for the calls to :obj:`step` in each sub-environment to finish.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait` times out. If ``None``, the call to :meth:`step_wait` never times out.

        Returns:
             The batched environment step information, (obs, reward, terminated, truncated, info)

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`step_wait` was called without any prior call to :meth:`step_async`.
            TimeoutError: If :meth:`step_wait` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(
                f"The call to `step_wait` has timed out after {timeout} second(s)."
            )

        observations, rewards, terminations, truncations, infos = [], [], [], [], {}
        successes = []
        for env_idx, pipe in enumerate(self.parent_pipes):
            env_step_return, success = pipe.recv()

            successes.append(success)
            if success:
                observations.append(env_step_return[0])
                rewards.append(env_step_return[1])
                terminations.append(env_step_return[2])
                truncations.append(env_step_return[3])
                infos = self._add_info(infos, env_step_return[4], env_idx)

        self._raise_if_errors(successes)

        if not self.shared_memory:
            self.observations = concatenate(
                self.single_observation_space,
                observations,
                self.observations,
            )

        self._state = AsyncState.DEFAULT
        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards, dtype=np.float64),
            np.array(terminations, dtype=np.bool_),
            np.array(truncations, dtype=np.bool_),
            infos,
        )

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Call a method from each parallel environment with args and kwargs.

        Args:
            name (str): Name of the method or property to call.
            *args: Position arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Returns:
            List of the results of the individual calls to the method or property for each environment.
        """
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def render(self) -> tuple[RenderFrame, ...] | None:
        """Returns a list of rendered frames from the environments."""
        return self.call("render")

    def call_async(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `call_async` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout: int | float | None = None) -> tuple[Any, ...]:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait` times out.
                If ``None`` (default), the call to :meth:`step_wait` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling :meth:`call_wait` without any prior call to :meth:`call_async`.
            TimeoutError: The call to :meth:`call_wait` has timed out after timeout second(s).
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def get_attr(self, name: str) -> tuple[Any, ...]:
        """Get a property from each parallel environment.

        Args:
            name (str): Name of the property to be get from each individual environment.

        Returns:
            The property with name
        """
        return self.call(name)

    def set_attr(self, name: str, values: list[Any] | tuple[Any] | object):
        """Sets an attribute of the sub-environments.

        Args:
            name: Name of the property to be set in each individual environment.
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
            AlreadyPendingCallError: Calling :meth:`set_attr` while waiting for a pending call to complete.
        """
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the number of environments. "
                f"Got `{len(values)}` values for {self.num_envs} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `set_attr` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def close_extras(self, timeout: int | float | None = None, terminate: bool = False):
        """Close the environments & clean up the extra resources (processes and pipes).

        Args:
            timeout: Number of seconds before the call to :meth:`close` times out. If ``None``,
                the call to :meth:`close` never times out. If the call to :meth:`close`
                times out, then all processes are terminated.
            terminate: If ``True``, then the :meth:`close` operation is forced and all processes are terminated.

        Raises:
            TimeoutError: If :meth:`close` timed out.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    f"Calling `close` while waiting for a pending call to `{self._state.value}` to complete."
                )
                function = getattr(self, f"{self._state.value}_wait")
                function(timeout)
        except multiprocessing.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll_pipe_envs(self, timeout: int | None = None):
        self._assert_is_running()

        if timeout is None:
            return True

        end_time = time.perf_counter() + timeout
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)

            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _check_spaces(self):
        self._assert_is_running()

        for pipe in self.parent_pipes:
            pipe.send(
                (
                    "_check_spaces",
                    (
                        self.observation_mode,
                        self.single_observation_space,
                        self.single_action_space,
                    ),
                )
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        same_observation_spaces, same_action_spaces = zip(*results)

        if not all(same_observation_spaces):
            if self.observation_mode == "same":
                raise RuntimeError(
                    "AsyncVectorEnv(..., observation_mode='same') however some of the sub-environments observation spaces are not equivalent. If this is intentional, use `observation_mode='different'` instead."
                )
            else:
                raise RuntimeError(
                    "AsyncVectorEnv(..., observation_mode='different' or custom space) however the sub-environment's observation spaces do not share a common shape and dtype."
                )

        if not all(same_action_spaces):
            raise RuntimeError(
                f"Some environments have an action space different from `{self.single_action_space}`. "
                "In order to batch actions, the action spaces from all environments must be equal."
            )

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                f"Trying to operate on `{type(self).__name__}`, after a call to `close()`."
            )

    def _raise_if_errors(self, successes: list[bool] | tuple[bool]):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value, trace = self.error_queue.get()

            logger.error(
                f"Received the following error from Worker-{index} - Shutting it down"
            )
            logger.error(f"{trace}")

            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                logger.error("Raising the last exception back to the main process.")
                self._state = AsyncState.DEFAULT
                raise exctype(value)

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)


def _async_worker(
    index: int,
    env_fn: Callable,
    pipe: Connection,
    parent_pipe: Connection,
    shared_memory: SynchronizedArray | dict[str, Any] | tuple[Any, ...],
    error_queue: Queue,
    autoreset_mode: AutoresetMode,
):
    env = env_fn()
    observation_space = env.observation_space
    action_space = env.action_space
    autoreset = False
    observation = None

    parent_pipe.close()

    try:
        while True:
            command, data = pipe.recv()

            if command == "reset":
                observation, info = env.reset(**data)
                if shared_memory:
                    write_to_shared_memory(
                        observation_space, index, observation, shared_memory
                    )
                    observation = None
                    autoreset = False
                pipe.send(((observation, info), True))
            elif command == "reset-noop":
                pipe.send(((observation, {}), True))
            elif command == "step":
                if autoreset_mode == AutoresetMode.NEXT_STEP:
                    if autoreset:
                        observation, info = env.reset()
                        reward, terminated, truncated = 0, False, False
                    else:
                        (
                            observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = env.step(data)
                    autoreset = terminated or truncated
                elif autoreset_mode == AutoresetMode.SAME_STEP:
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(data)

                    if terminated or truncated:
                        reset_observation, reset_info = env.reset()

                        info = {
                            "final_info": info,
                            "final_obs": observation,
                            **reset_info,
                        }
                        observation = reset_observation
                elif autoreset_mode == AutoresetMode.DISABLED:
                    assert autoreset is False
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(data)
                else:
                    raise ValueError(f"Unexpected autoreset_mode: {autoreset_mode}")

                if shared_memory:
                    write_to_shared_memory(
                        observation_space, index, observation, shared_memory
                    )
                    observation = None

                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "close", "_setattr", "_check_spaces"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with `call`, use `{name}` directly instead."
                    )

                attr = env.get_wrapper_attr(name)
                if callable(attr):
                    pipe.send((attr(*args, **kwargs), True))
                else:
                    pipe.send((attr, True))
            elif command == "_setattr":
                name, value = data
                env.set_wrapper_attr(name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                obs_mode, single_obs_space, single_action_space = data

                pipe.send(
                    (
                        (
                            (
                                single_obs_space == observation_space
                                if obs_mode == "same"
                                else is_space_dtype_shape_equiv(
                                    single_obs_space, observation_space
                                )
                            ),
                            single_action_space == action_space,
                        ),
                        True,
                    )
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must be one of [`reset`, `step`, `close`, `_call`, `_setattr`, `_check_spaces`]."
                )
    except (KeyboardInterrupt, Exception):
        error_type, error_message, _ = sys.exc_info()
        trace = traceback.format_exc()

        error_queue.put((index, error_type, error_message, trace))
        pipe.send((None, False))
    finally:
        env.close()
