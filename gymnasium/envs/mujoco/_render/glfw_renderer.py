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

"""An OpenGL renderer backed by GLFW."""

from gymnasium.envs.mujoco._render import base
from gymnasium.envs.mujoco._render import executor

# Re-raise any exceptions that occur during module import as `ImportError`s.
# This simplifies the conditional imports in `render/__init__.py`.
try:
  import glfw  # pylint: disable=g-import-not-at-top
except (ImportError, IOError, OSError) as exc:
  raise ImportError from exc
try:
  glfw.init()
except glfw.GLFWError as exc:
  raise ImportError from exc


class GLFWContext(base.ContextBase):
  """An OpenGL context backed by GLFW."""

  def __init__(self, max_width, max_height):
    # GLFWContext always uses `PassthroughRenderExecutor` rather than offloading
    # rendering calls to a separate thread because GLFW can only be safely used
    # from the main thread.
    super().__init__(max_width, max_height, executor.PassthroughRenderExecutor)

  def _platform_init(self, max_width, max_height):
    """Initializes this context.

    Args:
      max_width: Integer specifying the maximum framebuffer width in pixels.
      max_height: Integer specifying the maximum framebuffer height in pixels.
    """
    glfw.window_hint(glfw.VISIBLE, 0)
    glfw.window_hint(glfw.DOUBLEBUFFER, 0)
    self._context = glfw.create_window(width=max_width, height=max_height,
                                       title='Invisible window', monitor=None,
                                       share=None)
    # This reference prevents `glfw.destroy_window` from being garbage-collected
    # before the last window is destroyed, otherwise we may get
    # `AttributeError`s when the `__del__` method is later called.
    self._destroy_window = glfw.destroy_window

  def _platform_make_current(self):
    glfw.make_context_current(self._context)

  def _platform_free(self):
    """Frees resources associated with this context."""
    if self._context:
      if glfw.get_current_context() == self._context:
        glfw.make_context_current(None)
      self._destroy_window(self._context)
      self._context = None
