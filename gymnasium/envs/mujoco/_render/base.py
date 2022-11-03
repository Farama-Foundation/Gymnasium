# Copyright 2017 The dm_control Authors.
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

"""Base class for OpenGL context handlers.

`ContextBase` defines a common API that OpenGL rendering contexts should conform
to. In addition, it provides a `make_current` context manager that:

1. Makes this OpenGL context current within the appropriate rendering thread.
2. Yields an object exposing a `call` method that can be used to execute OpenGL
   calls within the rendering thread.

See the docstring for `dm_control.utils.render_executor` for further details
regarding rendering threads.
"""

import abc
import atexit
import collections
import contextlib
import sys
import weakref

from absl import logging
from gymnasium.envs.mujoco._render import executor

_CURRENT_CONTEXT_FOR_THREAD = collections.defaultdict(lambda: None)
_CURRENT_THREAD_FOR_CONTEXT = collections.defaultdict(lambda: None)


class ContextBase(metaclass=abc.ABCMeta):
  """Base class for managing OpenGL contexts."""

  def __init__(self,
               max_width,
               max_height,
               render_executor_class=executor.RenderExecutor):
    """Initializes this context."""
    logging.debug('Using render executor class: %s',
                  render_executor_class.__name__)
    self._render_executor = render_executor_class()
    self._refcount = 0

    self_weakref = weakref.ref(self)
    def _free_at_exit():
      if self_weakref():
        self_weakref()._free_unconditionally()  # pylint: disable=protected-access
    atexit.register(_free_at_exit)

    with self._render_executor.execution_context() as ctx:
      ctx.call(self._platform_init, max_width, max_height)

    self._patients = []

  def keep_alive(self, obj):
    self._patients.append(obj)

  def dont_keep_alive(self, obj):
    try:
      self._patients.remove(obj)
    except ValueError:
      pass

  def increment_refcount(self):
    self._refcount += 1

  def decrement_refcount(self):
    self._refcount -= 1

  @property
  def terminated(self):
    return self._render_executor.terminated

  @property
  def thread(self):
    return self._render_executor.thread

  def _free_on_executor_thread(self):  # pylint: disable=missing-function-docstring
    current_ctx = _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread]
    if current_ctx is not None:
      del _CURRENT_THREAD_FOR_CONTEXT[current_ctx]
    del _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread]

    self._platform_make_current()

    try:
      dummy = []
      while self._patients:
        patient = self._patients.pop()
        assert sys.getrefcount(patient) == sys.getrefcount(dummy)
        if hasattr(patient, 'free'):
          patient.free()
        del patient
    finally:
      self._platform_free()

  def free(self):
    """Frees resources associated with this context if its refcount is zero."""
    if self._refcount == 0:
      self._free_unconditionally()

  def _free_unconditionally(self):
    self._render_executor.terminate(self._free_on_executor_thread)

  def __del__(self):
    self._free_unconditionally()

  @contextlib.contextmanager
  def make_current(self):
    """Context manager that makes this Renderer's OpenGL context current.

    Yields:
      An object that exposes a `call` method that can be used to call a
      function on the dedicated rendering thread.

    Raises:
      RuntimeError: If this context is already current on another thread.
    """

    with self._render_executor.execution_context() as ctx:
      if _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread] != id(self):
        if _CURRENT_THREAD_FOR_CONTEXT[id(self)]:
          raise RuntimeError(
              'Cannot make context {!r} current on thread {!r}: '
              'this context is already current on another thread {!r}.'
              .format(self, self._render_executor.thread,
                      _CURRENT_THREAD_FOR_CONTEXT[id(self)]))
        else:
          current_context = (
              _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread])
          if current_context:
            del _CURRENT_THREAD_FOR_CONTEXT[current_context]
          _CURRENT_THREAD_FOR_CONTEXT[id(self)] = self._render_executor.thread
          _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread] = id(self)
          ctx.call(self._platform_make_current)
      yield ctx

  @abc.abstractmethod
  def _platform_init(self, max_width, max_height):
    """Performs an implementation-specific context initialization."""

  @abc.abstractmethod
  def _platform_make_current(self):
    """Make the OpenGL context current on the executing thread."""

  @abc.abstractmethod
  def _platform_free(self):
    """Performs an implementation-specific context cleanup."""
