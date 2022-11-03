# Copyright 2018 The dm_control Authors.
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

"""Tests for OSMesaContext."""

import unittest

from absl.testing import absltest
from dm_control import _render
import mock
from OpenGL import GL

MAX_WIDTH = 640
MAX_HEIGHT = 480

CONTEXT_PATH = _render.__name__ + '.pyopengl.osmesa_renderer.osmesa'
GL_ARRAYS_PATH = _render.__name__ + '.pyopengl.osmesa_renderer.arrays'


@unittest.skipUnless(
    _render.BACKEND == _render.constants.OSMESA,
    reason='OSMesa backend not selected.')
class OSMesaContextTest(absltest.TestCase):

  def test_init(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_osmesa:
      mock_osmesa.OSMesaCreateContextExt.return_value = mock_context
      renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
      self.assertIs(renderer._context, mock_context)
      renderer.free()

  def test_make_current(self):
    mock_context = mock.MagicMock()
    mock_buffer = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_osmesa:
      with mock.patch(GL_ARRAYS_PATH) as mock_glarrays:
        mock_osmesa.OSMesaCreateContextExt.return_value = mock_context
        mock_glarrays.GLfloatArray.zeros.return_value = mock_buffer
        renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
        with renderer.make_current():
          pass
        mock_osmesa.OSMesaMakeCurrent.assert_called_once_with(
            mock_context, mock_buffer, GL.GL_FLOAT, MAX_WIDTH, MAX_HEIGHT)
        renderer.free()

  def test_freeing(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_osmesa:
      mock_osmesa.OSMesaCreateContextExt.return_value = mock_context
      renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
      renderer.free()
      mock_osmesa.OSMesaDestroyContext.assert_called_once_with(mock_context)
      self.assertIsNone(renderer._context)


if __name__ == '__main__':
  absltest.main()
