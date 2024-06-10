"""This module implements various spaces.

Spaces describe mathematical sets and are used in Gym to specify valid actions and observations.
Every Gym environment must have the attributes ``action_space`` and ``observation_space``.
If, for instance, three possible actions (0,1,2) can be performed in your environment and observations
are vectors in the two-dimensional unit cube, the environment code may contain the following two lines::

    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(0, 1, shape=(2,))

All spaces inherit from the :class:`Space` superclass.
"""

from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.graph import Graph, GraphInstance
from gymnasium.spaces.multi_binary import MultiBinary
from gymnasium.spaces.multi_discrete import MultiDiscrete
from gymnasium.spaces.oneof import OneOf
from gymnasium.spaces.sequence import Sequence
from gymnasium.spaces.space import Space
from gymnasium.spaces.text import Text
from gymnasium.spaces.tuple import Tuple
from gymnasium.spaces.utils import flatdim, flatten, flatten_space, unflatten


__all__ = [
    # base space
    "Space",
    # fundamental spaces
    "Box",
    "Discrete",
    "Text",
    "MultiDiscrete",
    "MultiBinary",
    # composite spaces
    "Graph",
    "GraphInstance",
    "Tuple",
    "Sequence",
    "Dict",
    "OneOf",
    # util functions (there are more utility functions in vector/utils/spaces.py)
    "flatdim",
    "flatten_space",
    "flatten",
    "unflatten",
]
