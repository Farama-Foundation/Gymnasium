# Copyright 2017-2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""RenderExecutors executes OpenGL rendering calls on an appropriate thread.

The purpose of these classes is to ensure that OpenGL calls are made on the
same thread as where an OpenGL context was made current.

In a single-threaded setting, `PassthroughRenderExecutor` is essentially a no-op
that executes rendering calls on the same thread. This is provided to minimize
thread-switching overhead.

In a multithreaded setting, `OffloadingRenderExecutor` maintains a separate
dedicated thread on which the OpenGL context is created and made current. All
subsequent rendering calls are then offloaded onto this dedicated thread.
"""

import abc
import collections
from concurrent import futures
import contextlib
import threading


_NOT_IN_CONTEXT = 'Cannot be called outside of an `execution_context`.'
_ALREADY_TERMINATED = 'This executor has already been terminated.'


class _FakeLock:
  """An object with the same API as `threading.Lock` but that does nothing."""

  def acquire(self, blocking=True):
    pass

  def release(self):
    pass

  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback


_FAKE_LOCK = _FakeLock()


class BaseRenderExecutor(metaclass=abc.ABCMeta):
  """An object that manages rendering calls for an OpenGL context.

  This class helps ensure that OpenGL calls are made on the correct thread. The
  usage pattern is as follows:

  ```python
  executor = SomeRenderExecutorClass()
  with executor.execution_context() as ctx:
    ctx.call(an_opengl_call, arg, kwarg=foo)
    result = ctx.call(another_opengl_call)
  ```
  """

  def __init__(self):
    self._locked = 0
    self._terminated = False

  def _check_locked(self):
    if not self._locked:
      raise RuntimeError(_NOT_IN_CONTEXT)

  def _check_not_terminated(self):
    if self._terminated:
      raise RuntimeError(_ALREADY_TERMINATED)

  @contextlib.contextmanager
  def execution_context(self):
    """A context manager that allows calls to be offloaded to this executor."""
    self._check_not_terminated()
    with self._lock_if_necessary:
      self._locked += 1
      yield self
      self._locked -= 1

  @property
  def terminated(self):
    return self._terminated

  @property
  @abc.abstractmethod
  def thread(self):
    pass

  @property
  @abc.abstractmethod
  def _lock_if_necessary(self):
    pass

  @abc.abstractmethod
  def call(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def terminate(self, cleanup_callable=None):
    pass


class PassthroughRenderExecutor(BaseRenderExecutor):
  """A no-op render executor that executes on the calling thread."""

  def __init__(self):
    super().__init__()
    self._mutex = threading.RLock()

  @property
  def thread(self):
    if not self._terminated:
      return threading.current_thread()
    else:
      return None

  @property
  def _lock_if_necessary(self):
    return self._mutex

  def call(self, func, *args, **kwargs):
    self._check_locked()
    return func(*args, **kwargs)

  def terminate(self, cleanup_callable=None):
    with self._lock_if_necessary:
      if not self._terminated:
        if cleanup_callable:
          cleanup_callable()
        self._terminated = True


class _ThreadPoolExecutorPool:
  """A pool of reusable ThreadPoolExecutors."""

  def __init__(self):
    self._deque = collections.deque()
    self._lock = threading.Lock()

  def acquire(self):
    with self._lock:
      if self._deque:
        return self._deque.popleft()
      else:
        return futures.ThreadPoolExecutor(max_workers=1)

  def release(self, thread_pool_executor):
    with self._lock:
      self._deque.append(thread_pool_executor)


_THREAD_POOL_EXECUTOR_POOL = _ThreadPoolExecutorPool()


class OffloadingRenderExecutor(BaseRenderExecutor):
  """A render executor that executes calls on a dedicated offload thread."""

  def __init__(self):
    super().__init__()
    self._mutex = threading.RLock()
    self._executor = _THREAD_POOL_EXECUTOR_POOL.acquire()
    self._thread = self._executor.submit(threading.current_thread).result()

  @property
  def thread(self):
    return self._thread

  @property
  def _lock_if_necessary(self):
    if threading.current_thread() is self.thread:
      # If the offload thread needs to make a call to its own executor, for
      # example when a weakref callback is triggered during an offloaded call,
      # then we must not try to reacquire our own lock.
      # Otherwise, a deadlock ensues.
      return _FAKE_LOCK
    else:
      return self._mutex

  def call(self, func, *args, **kwargs):
    self._check_locked()
    return self._call_locked(func, *args, **kwargs)

  def _call_locked(self, func, *args, **kwargs):
    if threading.current_thread() is self.thread:
      # If the offload thread needs to make a call to its own executor, for
      # example when a weakref callback is triggered during an offloaded call,
      # we should just directly call the function.
      # Otherwise, a deadlock ensues.
      return func(*args, **kwargs)
    else:
      return self._executor.submit(func, *args, **kwargs).result()

  def terminate(self, cleanup_callable=None):
    if self._terminated:
      return
    with self._lock_if_necessary:
      if not self._terminated:
        if cleanup_callable:
          self._call_locked(cleanup_callable)
        _THREAD_POOL_EXECUTOR_POOL.release(self._executor)
        self._executor = None
        self._thread = None
        self._terminated = True
