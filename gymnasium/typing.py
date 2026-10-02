r"""Public typing vocabulary shared by Gymnasium's generic classes.

This module is the single, centralised source of the :class:`~typing.TypeVar`\\ s
used to parameterise Gymnasium's generic classes (precedent: :mod:`numpy.typing`).
Downstream code and Gymnasium's own modules should import these names from here
rather than redefining their own copies.

.. warning::
    These TypeVars are **provisional**. They exist because Gymnasium supports Python
    versions without PEP 695 type parameter syntax, and will be replaced by that syntax
    when support for those versions is dropped. Do not build long-lived abstractions on them.

Every TypeVar declares ``default=Any`` (PEP 696), so a class may be subscripted with
as few or as many arguments as desired and any omitted argument falls back to ``Any``.

The single-environment vocabulary parameterises
:class:`gymnasium.Env` ``[ObsType, ActType]`` and
:class:`gymnasium.Wrapper` ``[WrapperObsType, WrapperActType, ObsType, ActType]``;
these are invariant for backwards compatibility.
The vector-environment vocabulary parameterises
:class:`gymnasium.vector.VectorEnv` ``[VectorObsType_co, VectorActType_contra, VectorRewardType_co, VectorBoolType_co]``
and its wrappers; these are variance-correct, as indicated by their ``_co`` (covariant)
and ``_contra`` (contravariant) suffixes, such that PEP 695's inferred variance will
not change their semantics. Each name's meaning is documented on the name itself below.
"""

from typing import Any, TypeAlias

import numpy as np
from typing_extensions import TypeVar

__all__ = [
    # Single Agent Env
    "ObsType",
    "ActType",
    "RenderFrame",
    "WrapperObsType",
    "WrapperActType",
    # Vector Env
    "VectorObsType_co",
    "VectorActType_contra",
    "VectorRewardType_co",
    "VectorBoolType_co",
    "VectorWrappedObsType",
    "VectorWrappedActType",
    "VectorWrappedRewardType",
    # Deprecated
    "ArrayType",
]

# Single-environment vocabulary
ObsType = TypeVar("ObsType", default=Any)
"""The observation type of an :class:`~gymnasium.Env`, i.e. what :meth:`~gymnasium.Env.reset` and :meth:`~gymnasium.Env.step` return and :attr:`~gymnasium.Env.observation_space` contains."""

ActType = TypeVar("ActType", default=Any)
"""The action type of an :class:`~gymnasium.Env`, i.e. what :meth:`~gymnasium.Env.step` accepts and :attr:`~gymnasium.Env.action_space` contains."""

RenderFrame: TypeAlias = str | np.ndarray | tuple[np.ndarray, np.ndarray]
"""A single frame returned by :meth:`~gymnasium.Env.render` (a concrete alias, not a TypeVar)."""

WrapperObsType = TypeVar("WrapperObsType", default=Any)
"""The observation type a :class:`~gymnasium.Wrapper` exposes to its user, possibly different from the wrapped environment's :data:`ObsType`."""

WrapperActType = TypeVar("WrapperActType", default=Any)
"""The action type a :class:`~gymnasium.Wrapper` accepts from its user, possibly different from the wrapped environment's :data:`ActType`."""

# Vector-environment vocabulary
VectorObsType_co = TypeVar("VectorObsType_co", covariant=True, default=Any)
"""The batched observation type of a :class:`~gymnasium.vector.VectorEnv` (covariant)."""

VectorActType_contra = TypeVar("VectorActType_contra", contravariant=True, default=Any)
"""The batched action type of a :class:`~gymnasium.vector.VectorEnv` (contravariant)."""

VectorRewardType_co = TypeVar("VectorRewardType_co", covariant=True, default=Any)
"""The batched reward array type of a :class:`~gymnasium.vector.VectorEnv`, typically ``np.ndarray`` of ``float64`` (covariant)."""

VectorBoolType_co = TypeVar("VectorBoolType_co", covariant=True, default=Any)
"""The batched termination/truncation array type of a :class:`~gymnasium.vector.VectorEnv`, typically ``np.ndarray`` of ``bool`` (covariant)."""

# `Wrapped` variants are the wrapped (inner) environment's types for wrappers that
# transform observations, actions or rewards. They default to the wrapper's own type
# so that a same-type wrapper doesn't need to repeat itself.
#
# Ordering constraint: a TypeVar whose default refers to another TypeVar is only valid
# when that other TypeVar precedes it in *every* parameter list it appears in, e.g.
# `VectorObsType_co` must come before `VectorWrappedObsType` in `Generic[...]`.
# Otherwise, type checkers report a default that refers to type variables that are out
# of scope. As these TypeVars are shared across modules, this applies to every generic
# class in Gymnasium that uses them.
VectorWrappedObsType = TypeVar("VectorWrappedObsType", default=VectorObsType_co)
VectorWrappedActType = TypeVar("VectorWrappedActType", default=VectorActType_contra)
VectorWrappedRewardType = TypeVar(
    "VectorWrappedRewardType", default=VectorRewardType_co
)

# Deprecated: kept for backwards compatibility with downstream code that does
# `from gymnasium.vector.vector_env import ArrayType`. Prefer the dedicated
# reward/bool array TypeVars above.
ArrayType = TypeVar("ArrayType", default=Any)
