"""Scenario: the numerical integrator and the closed form must tell the same story."""

import math

import numpy as np
import pytest

from corruption.model import (
    BearerParams,
    Fate,
    SolverError,
    analytic_fall_time,
    analytic_solution,
    fate,
    simulate,
)

rng = np.random.default_rng(20261023)
RANDOM_BEARERS = [
    BearerParams(
        W=float(rng.uniform(0, 1)),
        A=float(rng.uniform(0, 1)),
        beta=float(rng.uniform(0, 5)),
        c0=float(rng.uniform(0, 0.3)),
    )
    for _ in range(40)
]


@pytest.mark.parametrize("p", RANDOM_BEARERS)
def test_numeric_fall_time_matches_closed_form(p: BearerParams) -> None:
    traj = simulate(p, t_max=2000.0)
    expected = analytic_fall_time(p)
    if math.isinf(expected) or expected > 2000.0:
        assert not traj.fell
    else:
        assert traj.fall_time == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("p", RANDOM_BEARERS[:10])
def test_numeric_trajectory_matches_closed_form(p: BearerParams) -> None:
    traj = simulate(p, t_max=300.0)
    before_fall = traj.t < traj.fall_time
    expected = analytic_solution(p, traj.t[before_fall])
    np.testing.assert_allclose(traj.c[before_fall], expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("p", RANDOM_BEARERS)
def test_fate_classification_agrees_with_long_simulation(p: BearerParams) -> None:
    traj = simulate(p, t_max=1e5, n_points=10)
    assert traj.fell == (fate(p) is Fate.FALLS)


def test_bearer_sitting_on_the_point_of_no_return_is_not_misreported() -> None:
    # Floating point puts the bearer a hair on either side; the closed form must
    # report it as balanced or resisting, never as an early fall.
    p = BearerParams(W=0.9, A=0.3, c0=1 / 3)
    assert fate(p) is not Fate.FALLS or analytic_fall_time(p) > 100.0


def test_exploding_usage_surfaces_as_an_error_not_a_silent_result() -> None:
    p = BearerParams(W=0.5, A=0.5)

    def broken(t: float) -> float:
        return math.nan if t > 1.0 else 1.0

    with pytest.raises((SolverError, ValueError)):
        simulate(p, t_max=50.0, usage=broken)
