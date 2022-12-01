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

"""Extends OpenGL.EGL with definitions necessary for headless rendering."""

import ctypes
from OpenGL.platform import ctypesloader  # pylint: disable=g-bad-import-order
try:
  # Nvidia driver seems to need libOpenGL.so (as opposed to libGL.so)
  # for multithreading to work properly. We load this in before everything else.
  ctypesloader.loadLibrary(ctypes.cdll, 'OpenGL', mode=ctypes.RTLD_GLOBAL)
except OSError:
  pass

# pylint: disable=g-import-not-at-top

from OpenGL import EGL
from OpenGL import error


# From the EGL_EXT_device_enumeration extension.
PFNEGLQUERYDEVICESEXTPROC = ctypes.CFUNCTYPE(
    EGL.EGLBoolean,
    EGL.EGLint,
    ctypes.POINTER(EGL.EGLDeviceEXT),
    ctypes.POINTER(EGL.EGLint),
)
try:
  _eglQueryDevicesEXT = PFNEGLQUERYDEVICESEXTPROC(  # pylint: disable=invalid-name
      EGL.eglGetProcAddress('eglQueryDevicesEXT'))
except TypeError:
  raise ImportError('eglQueryDevicesEXT is not available.')


# From the EGL_EXT_platform_device extension.
EGL_PLATFORM_DEVICE_EXT = 0x313F
PFNEGLGETPLATFORMDISPLAYEXTPROC = ctypes.CFUNCTYPE(
    EGL.EGLDisplay, EGL.EGLenum, ctypes.c_void_p, ctypes.POINTER(EGL.EGLint))
try:
  eglGetPlatformDisplayEXT = PFNEGLGETPLATFORMDISPLAYEXTPROC(  # pylint: disable=invalid-name
      EGL.eglGetProcAddress('eglGetPlatformDisplayEXT'))
except TypeError:
  raise ImportError('eglGetPlatformDisplayEXT is not available.')


# Wrap raw _eglQueryDevicesEXT function into something more Pythonic.
def eglQueryDevicesEXT(max_devices=10):  # pylint: disable=invalid-name
  devices = (EGL.EGLDeviceEXT * max_devices)()
  num_devices = EGL.EGLint()
  success = _eglQueryDevicesEXT(max_devices, devices, num_devices)
  if success == EGL.EGL_TRUE:
    return [devices[i] for i in range(num_devices.value)]
  else:
    raise error.GLError(err=EGL.eglGetError(),
                        baseOperation=eglQueryDevicesEXT,
                        result=success)


# Expose everything from upstream so that
# we can use this as a drop-in replacement for OpenGL.EGL.
# pylint: disable=wildcard-import,g-bad-import-order
from OpenGL.EGL import *
