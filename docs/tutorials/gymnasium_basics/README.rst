Gymnasium Basics
================

.. _gallery_section_name:

The aim of these tutorials is to showcase the fundamental API of Gymnasium to help users implement it.

Each tutorial is a self-contained Python script: its page shows the code next to the output, and the
script can be downloaded from the same page to run and modify locally.

What the tutorials cover
------------------------

- **Make your own custom environment** - generate a new environment from the Farama
  ``gymnasium-env-template`` with Copier, then implement its spaces, transitions, rendering,
  registration and tests.
- **Implementing custom wrappers** - subclass ``ObservationWrapper``, ``ActionWrapper``,
  ``RewardWrapper`` and ``Wrapper`` to change the observations, actions or rewards of an existing
  environment.
- **Handling time limits** - the difference between ``terminated`` and ``truncated``, and why
  bootstrapping should stop only on ``terminated``.
- **Load custom quadruped robot environments** - pass an MJCF or URDF model to a MuJoCo environment
  with the ``xml_file`` argument, then tune its simulation, termination, reward and observation
  parameters.

Requirements
------------

The tutorials require Python 3.11 or newer with ``gymnasium`` installed:

.. code:: console

    pip install gymnasium

Three of the tutorials also use optional dependencies:

- ``pip install copier "gymnasium[classic-control]"`` for *Make your own custom environment*, which
  walks through the Copier template and renders the example environment with ``pygame``.
- ``pip install "gymnasium[box2d]"`` for *Implementing custom wrappers*, which wraps
  ``LunarLanderContinuous-v3``.
- ``pip install "gymnasium[mujoco]"`` for *Load custom quadruped robot environments*, which wraps
  ``Ant-v5``.

Running a tutorial locally
--------------------------

Every tutorial page ends with a *Download Python source code* link. Save the file and run it from a
terminal, for example:

.. code:: console

    python implementing_custom_wrappers.py

The tutorials are written to be read as much as run: the printed output and figures are part of the
explanation, so the rendered page is the primary reference and the script is there to experiment
with.

Where to go next
----------------

- `Basic usage <https://gymnasium.farama.org/introduction/basic_usage/>`_ introduces the
  ``reset`` / ``step`` loop, spaces and wrappers that every tutorial builds on.
- The `API reference <https://gymnasium.farama.org/api/env/>`_ documents the environment API in
  full.
