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

"""RenderExecutor executes OpenGL rendering calls on an appropriate thread.

OpenGL calls must be made on the same thread as where an OpenGL context is
made current on. With GPU rendering, migrating OpenGL contexts between threads
can become expensive. We provide a thread-safe executor that maintains a
thread on which an OpenGL context can be kept permanently current, and any other
threads that wish to render with that context will have their rendering calls
offloaded to the dedicated thread.

For single-threaded applications, set the `DISABLE_RENDER_THREAD_OFFLOADING`
environment variable before launching the Python interpreter. This will
eliminate the overhead of unnecessary thread-switching.
"""

# pylint: disable=g-import-not-at-top
import os
_OFFLOAD = not bool(os.environ.get('DISABLE_RENDER_THREAD_OFFLOADING', ''))
del os

from gymnasium.envs.mujoco._render.executor.render_executor import BaseRenderExecutor
from gymnasium.envs.mujoco._render.executor.render_executor import OffloadingRenderExecutor
from gymnasium.envs.mujoco._render.executor.render_executor import PassthroughRenderExecutor

_EXECUTORS = (PassthroughRenderExecutor, OffloadingRenderExecutor)

try:
  from gymnasium.envs.mujoco._render.executor.native_mutex.render_executor import NativeMutexOffloadingRenderExecutor
  _EXECUTORS += (NativeMutexOffloadingRenderExecutor,)
except ImportError:
  NativeMutexOffloadingRenderExecutor = None

if _OFFLOAD:
  RenderExecutor = (  # pylint: disable=invalid-name
      NativeMutexOffloadingRenderExecutor or OffloadingRenderExecutor)
else:
  RenderExecutor = PassthroughRenderExecutor  # pylint: disable=invalid-name
