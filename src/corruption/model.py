"""Deterministic corruption model.

Corruption C of a ring-bearer evolves as

    dC/dt = α·u(t)·A·(1 + β·C) − γ·W·(1 − C)

with will W, ambition A and ring usage u(t). The bearer falls (becomes a wraith)
when C reaches the threshold C*.

For constant usage u the right-hand side is linear in C:

    dC/dt = a·C + b,   a = α·u·A·β + γ·W,   b = α·u·A − γ·W

so the model has a single fixed point C_u = −b/a. Because a > 0 it is unstable:
it is the point of no return. Below it corruption drains to zero (the bearer
resists forever); above it corruption grows until the fall.

This module is a pure function of (W, A, α, β, γ, u(t)): no plotting, no I/O,
so it can be ported to other languages as-is.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

Usage = Callable[[float], float]
FloatArray = NDArray[np.float64]


class ParameterError(ValueError):
    """Raised when model parameters are outside their valid domain."""


class SolverError(RuntimeError):
    """Raised when the numerical integrator fails."""


class Fate(Enum):
    """Long-run outcome of a bearer under constant usage."""

    FALLS = "falls"
    RESISTS = "resists"
    BALANCED = "balanced"


@dataclass(frozen=True)
class BearerParams:
    """Parameters of one bearer.

    Attributes:
        W: will, the strength that drains corruption (≥ 0).
        A: ambition, how strongly ring use feeds corruption (≥ 0).
        alpha: corruption rate per unit of usage and ambition (> 0).
        beta: positive feedback, how much existing corruption accelerates growth (≥ 0).
        gamma: recovery rate per unit of will (≥ 0).
        c_star: fall threshold, 0 < C* ≤ 1.
        c0: corruption when the ring is received, 0 ≤ C0 < C*.
    """

    W: float
    A: float
    alpha: float = 0.02
    beta: float = 3.0
    gamma: float = 0.02
    c_star: float = 0.9
    c0: float = 0.05

    def __post_init__(self) -> None:
        for name in ("W", "A", "alpha", "beta", "gamma", "c_star", "c0"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ParameterError(f"{name} must be finite, got {value!r}")
        if self.W < 0 or self.A < 0:
            raise ParameterError("W and A must be non-negative")
        if self.alpha <= 0:
            raise ParameterError("alpha must be positive")
        if self.beta < 0 or self.gamma < 0:
            raise ParameterError("beta and gamma must be non-negative")
        if not 0 < self.c_star <= 1:
            raise ParameterError("c_star must be in (0, 1]")
        if not 0 <= self.c0 < self.c_star:
            raise ParameterError("c0 must be in [0, c_star)")


@dataclass(frozen=True)
class Trajectory:
    """Solution of the ODE up to the fall or the horizon.

    `fall_time` is `math.inf` when the bearer did not fall before the horizon.
    """

    t: FloatArray
    c: FloatArray
    fall_time: float

    @property
    def fell(self) -> bool:
        return math.isfinite(self.fall_time)


def constant_usage(u: float) -> Usage:
    """Usage profile that wears the ring at a constant intensity `u` ≥ 0."""
    if not math.isfinite(u) or u < 0:
        raise ParameterError("usage must be finite and non-negative")
    return lambda _t: u


def rhs(t: float, c: float, p: BearerParams, usage: Usage) -> float:
    """Right-hand side dC/dt, clamped so corruption never drops below zero."""
    u = usage(t)
    if not math.isfinite(u) or u < 0:
        raise ParameterError(f"usage returned {u!r} at t={t}; it must be finite and ≥ 0")
    drift = p.alpha * u * p.A * (1.0 + p.beta * c) - p.gamma * p.W * (1.0 - c)
    if c <= 0.0 and drift < 0.0:
        return 0.0
    return drift


def drift_coefficients(p: BearerParams, u: float = 1.0) -> tuple[float, float]:
    """Coefficients (a, b) of the linear form dC/dt = a·C + b for constant usage."""
    a = p.alpha * u * p.A * p.beta + p.gamma * p.W
    b = p.alpha * u * p.A - p.gamma * p.W
    return a, b


def unstable_fixed_point(p: BearerParams, u: float = 1.0) -> float:
    """Point of no return C_u = (γW − αuA) / (αuAβ + γW).

    Returns `-inf` when nothing holds corruption back (a = 0, b > 0) and `nan`
    when every C is an equilibrium (a = b = 0).
    """
    a, b = drift_coefficients(p, u)
    if a == 0.0:
        return -math.inf if b > 0 else math.nan
    return -b / a


def critical_ambition(p: BearerParams, u: float = 1.0) -> float:
    """Ambition A_c = γW / (αu) at which the point of no return reaches C = 0.

    Above A_c even an uncorrupted bearer eventually falls. A_c does not depend on β.
    """
    if u <= 0:
        return math.inf
    return p.gamma * p.W / (p.alpha * u)


def fate(p: BearerParams, u: float = 1.0) -> Fate:
    """Long-run fate under constant usage, from the position of C0 against C_u."""
    a, b = drift_coefficients(p, u)
    if a == 0.0:
        return Fate.FALLS if b > 0 else Fate.BALANCED
    c_u = -b / a
    if p.c0 > c_u:
        return Fate.FALLS
    if p.c0 < c_u:
        return Fate.RESISTS
    return Fate.BALANCED


def analytic_solution(p: BearerParams, t: FloatArray | float, u: float = 1.0) -> FloatArray:
    """Closed-form C(t) for constant usage, clamped at zero like `rhs`.

    Values past the fall are returned unclamped above; callers stop at C*.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    a, b = drift_coefficients(p, u)
    if a == 0.0:
        c = p.c0 + b * t_arr
    else:
        c_u = -b / a
        c = c_u + (p.c0 - c_u) * np.exp(a * t_arr)
    return np.maximum(c, 0.0)


def analytic_fall_time(p: BearerParams, u: float = 1.0) -> float:
    """Closed-form time at which C reaches C*, or `inf` if the bearer never falls."""
    if fate(p, u) is not Fate.FALLS:
        return math.inf
    a, b = drift_coefficients(p, u)
    if a == 0.0:
        return (p.c_star - p.c0) / b
    c_u = -b / a
    return math.log((p.c_star - c_u) / (p.c0 - c_u)) / a


def simulate(
    p: BearerParams,
    t_max: float,
    usage: Usage | None = None,
    *,
    n_points: int = 500,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = math.inf,
) -> Trajectory:
    """Integrate the ODE with `solve_ivp` until the fall (C = C*) or `t_max`.

    Args:
        p: bearer parameters.
        t_max: horizon (> 0).
        usage: u(t); defaults to wearing the ring constantly (u = 1).
        n_points: number of output samples on [0, t_max] (truncated at the fall).
        rtol, atol: integrator tolerances.
        max_step: cap on the step size; set it below the shortest feature of u(t).
    """
    if not math.isfinite(t_max) or t_max <= 0:
        raise ParameterError("t_max must be finite and positive")
    if n_points < 2:
        raise ParameterError("n_points must be at least 2")
    u_fn = usage if usage is not None else constant_usage(1.0)

    def fall_event(t: float, y: FloatArray) -> float:
        return float(y[0]) - p.c_star

    fall_event.terminal = True  # type: ignore[attr-defined]
    fall_event.direction = 1.0  # type: ignore[attr-defined]

    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(
        lambda t, y: [rhs(t, float(y[0]), p, u_fn)],
        (0.0, t_max),
        [p.c0],
        t_eval=t_eval,
        events=fall_event,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if sol.status == -1:
        raise SolverError(sol.message)

    t_out = np.asarray(sol.t, dtype=np.float64)
    c_out = np.maximum(np.asarray(sol.y[0], dtype=np.float64), 0.0)
    fall_time = math.inf
    if sol.t_events[0].size:
        fall_time = float(sol.t_events[0][0])
        t_out = np.append(t_out, fall_time)
        c_out = np.append(c_out, p.c_star)
    return Trajectory(t=t_out, c=c_out, fall_time=fall_time)
