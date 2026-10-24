"""Scenario: the vectorized sweep and the scalar ODE must agree on every grid cell."""

import math

import numpy as np
import pytest

from corruption.bifurcation import sweep
from corruption.model import BearerParams, analytic_fall_time, simulate, unstable_fixed_point

A_AXIS = np.linspace(0.05, 1.5, 12)
BETA_AXIS = np.linspace(0.0, 6.0, 5)


@pytest.mark.parametrize("W", [0.3, 0.6, 0.9])
def test_sweep_matches_scalar_closed_form_on_every_cell(W: float) -> None:
    base = BearerParams(W=W, A=0.0, c0=0.05)
    result = sweep(base, A_AXIS, BETA_AXIS)
    for i, beta in enumerate(BETA_AXIS):
        for j, A in enumerate(A_AXIS):
            p = BearerParams(W=W, A=float(A), beta=float(beta), c0=0.05)
            assert result.fixed_point[i, j] == pytest.approx(unstable_fixed_point(p))
            expected = analytic_fall_time(p)
            if math.isinf(expected):
                assert math.isinf(result.fall_time[i, j])
            else:
                assert result.fall_time[i, j] == pytest.approx(expected)


def test_sweep_fate_matches_long_ode_runs() -> None:
    base = BearerParams(W=0.6, A=0.0, c0=0.05)
    result = sweep(base, A_AXIS, BETA_AXIS)
    for i, beta in enumerate(BETA_AXIS):
        for j, A in enumerate(A_AXIS):
            p = BearerParams(W=0.6, A=float(A), beta=float(beta), c0=0.05)
            traj = simulate(p, t_max=1e5, n_points=5)
            assert traj.fell == bool(result.falls[i, j]), (A, beta)
