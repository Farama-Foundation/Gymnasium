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

"""String constants for the rendering module."""

# Name of the environment variable that selects a renderer platform.
MUJOCO_GL = 'MUJOCO_GL'

# Name of the environment variable that selects a platform for PyOpenGL.
PYOPENGL_PLATFORM = 'PYOPENGL_PLATFORM'

# Renderer platform specifiers.
# All values in each tuple are synonyms for the MUJOCO_GL environment variable.
# The first entry in each tuple is considered "canonical", and is the one
# assigned to the _render.BACKEND variable.
OSMESA = ('osmesa',)
GLFW = ('glfw', 'on', 'enable', 'enabled', 'true', '1', '')
EGL = ('egl',)
NO_RENDERER = ('off', 'disable', 'disabled', 'false', '0')

