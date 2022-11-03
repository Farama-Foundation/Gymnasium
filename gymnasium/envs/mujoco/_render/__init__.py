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

"""OpenGL context management for rendering MuJoCo scenes.

By default, the `Renderer` class will try to load one of the following rendering
APIs, in descending order of priority: GLFW > EGL > OSMesa.

It is also possible to select a specific backend by setting the `MUJOCO_GL=`
environment variable to 'glfw', 'egl', or 'osmesa'.
"""

import collections
import os

from absl import logging
from gymnasium.envs.mujoco._render import constants

BACKEND = os.environ.get(constants.MUJOCO_GL)


# pylint: disable=g-import-not-at-top
def _import_egl():
  from gymnasium.envs.mujoco._render.pyopengl.egl_renderer import EGLContext
  return EGLContext


def _import_glfw():
  from gymnasium.envs.mujoco._render.glfw_renderer import GLFWContext
  return GLFWContext


def _import_osmesa():
  from gymnasium.envs.mujoco._render.pyopengl.osmesa_renderer import OSMesaContext
  return OSMesaContext
# pylint: enable=g-import-not-at-top


def _no_renderer():
  def no_renderer(*args, **kwargs):
    del args, kwargs
    raise RuntimeError('No OpenGL rendering backend is available.')
  return no_renderer


_ALL_RENDERERS = (
    (constants.GLFW, _import_glfw),
    (constants.EGL, _import_egl),
    (constants.OSMESA, _import_osmesa),
)

_NO_RENDERER = (
    (constants.NO_RENDERER, _no_renderer),
)


if BACKEND is not None:
  # If a backend was specified, try importing it and error if unsuccessful.
  import_func = None
  for names, importer in _ALL_RENDERERS + _NO_RENDERER:
    if BACKEND in names:
      import_func = importer
      BACKEND = names[0]  # canonicalize the renderer name
      break
  if import_func is None:
    all_names = set()
    for names, _ in _ALL_RENDERERS + _NO_RENDERER:
      all_names.update(names)
    raise RuntimeError(
        'Environment variable {} must be one of {!r}: got {!r}.'
        .format(constants.MUJOCO_GL, sorted(all_names), BACKEND))
  logging.info('MUJOCO_GL=%s, attempting to import specified OpenGL backend.',
               BACKEND)
  Renderer = import_func()
else:
  logging.info('MUJOCO_GL is not set, so an OpenGL backend will be chosen '
               'automatically.')
  # Otherwise try importing them in descending order of priority until
  # successful.
  for names, import_func in _ALL_RENDERERS:
    try:
      Renderer = import_func()
      BACKEND = names[0]
      logging.info('Successfully imported OpenGL backend: %s', names[0])
      break
    except ImportError:
      logging.info('Failed to import OpenGL backend: %s', names[0])
  if BACKEND is None:
    logging.info('No OpenGL backend could be imported. Attempting to create a '
                 'rendering context will result in a RuntimeError.')
    Renderer = _no_renderer()

USING_GPU = BACKEND in constants.EGL + constants.GLFW
