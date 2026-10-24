"""Parameter sweeps and the fixed-point structure of the corruption model.

Under constant usage the model has two candidate equilibria in [0, 1]:

- C = 0, held by the clamp. It is stable while the drift at zero is negative,
  that is while A < A_c = γW / (αu).
- C_u = (γW − αuA) / (αuAβ + γW), the unstable point of no return. It lies in
  [0, 1) exactly when A ≤ A_c and leaves through C = 0 at A = A_c.

So A_c is a bifurcation point: below it a resisting state exists, above it every
bearer falls. A_c does not depend on β. A bearer who starts with C0 > 0 falls
once C0 > C_u, which happens above the fate boundary

    A_f(β) = γW(1 − C0) / (αu(1 + βC0)),

and that boundary does depend on β: feedback only matters for bearers who are
already partly corrupted when they take the ring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from corruption.model import BearerParams, FloatArray, ParameterError, critical_ambition

__all__ = [
    "Sweep",
    "critical_ambition",
    "fate_boundary_ambition",
    "fixed_point",
    "sweep",
]

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Sweep:
    """Result of a sweep over ambition A (columns) and feedback β (rows).

    `fixed_point`, `falls` and `fall_time` have shape (len(beta), len(A)).
    `fall_time` is `inf` where the bearer resists.
    """

    A: FloatArray
    beta: FloatArray
    fixed_point: FloatArray
    falls: BoolArray
    fall_time: FloatArray


def _as_grid_axis(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ParameterError(f"{name} must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ParameterError(f"{name} values must be finite and non-negative")
    return arr


def fixed_point(base: BearerParams, A: ArrayLike, beta: ArrayLike, u: float = 1.0) -> FloatArray:
    """Vectorized C_u over broadcastable A and β; `-inf` when nothing restores (a = 0)."""
    A_arr = np.asarray(A, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    drive = base.alpha * u * A_arr
    restore = base.gamma * base.W
    a = drive * beta_arr + restore
    b = drive - restore
    with np.errstate(divide="ignore", invalid="ignore"):
        c_u = np.where(a > 0, -b / np.where(a > 0, a, 1.0), np.where(b > 0, -np.inf, np.nan))
    return np.asarray(c_u, dtype=np.float64)


def fate_boundary_ambition(base: BearerParams, beta: ArrayLike, u: float = 1.0) -> FloatArray:
    """A_f(β) = γW(1 − C0) / (αu(1 + βC0)): above it a bearer starting at C0 falls."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    if u <= 0:
        return np.full_like(beta_arr, np.inf)
    return np.asarray(
        base.gamma * base.W * (1 - base.c0) / (base.alpha * u * (1 + beta_arr * base.c0)),
        dtype=np.float64,
    )


def sweep(
    base: BearerParams,
    A_values: ArrayLike,
    beta_values: ArrayLike,
    u: float = 1.0,
) -> Sweep:
    """Fixed point, fate and closed-form fall time on an (β × A) grid.

    W, α, γ, C0 and C* come from `base`; its own A and β are ignored.
    """
    if not np.isfinite(u) or u < 0:
        raise ParameterError("usage must be finite and non-negative")
    A_axis = _as_grid_axis(A_values, "A_values")
    beta_axis = _as_grid_axis(beta_values, "beta_values")
    A_grid, beta_grid = np.meshgrid(A_axis, beta_axis)

    drive = base.alpha * u * A_grid
    restore = base.gamma * base.W
    a = drive * beta_grid + restore
    b = drive - restore
    c_u = fixed_point(base, A_grid, beta_grid, u)
    falls = np.where(a > 0, base.c0 > c_u, b > 0)

    fall_time = np.full(A_grid.shape, np.inf)
    exp_mask = falls & (a > 0)
    fall_time[exp_mask] = (
        np.log((base.c_star - c_u[exp_mask]) / (base.c0 - c_u[exp_mask])) / a[exp_mask]
    )
    lin_mask = falls & (a == 0)
    fall_time[lin_mask] = (base.c_star - base.c0) / b[lin_mask]

    return Sweep(A=A_axis, beta=beta_axis, fixed_point=c_u, falls=falls, fall_time=fall_time)
