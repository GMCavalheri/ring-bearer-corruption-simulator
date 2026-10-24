import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from corruption.model import (
    BearerParams,
    Fate,
    ParameterError,
    analytic_fall_time,
    analytic_solution,
    constant_usage,
    critical_ambition,
    drift_coefficients,
    fate,
    rhs,
    simulate,
    unstable_fixed_point,
)

FALLER = BearerParams(W=0.5, A=0.8)
RESISTER = BearerParams(W=0.9, A=0.3)


def test_rhs_matches_the_formula() -> None:
    p = BearerParams(W=0.4, A=0.7, alpha=0.03, beta=2.0, gamma=0.05)
    c = 0.3
    expected = 0.03 * 0.5 * 0.7 * (1 + 2.0 * c) - 0.05 * 0.4 * (1 - c)
    assert rhs(0.0, c, p, constant_usage(0.5)) == pytest.approx(expected)


def test_rhs_is_linear_in_c_with_the_documented_coefficients() -> None:
    p = BearerParams(W=0.4, A=0.7)
    a, b = drift_coefficients(p, 1.0)
    for c in (0.1, 0.5, 0.8):
        assert rhs(0.0, c, p, constant_usage(1.0)) == pytest.approx(a * c + b)


def test_rhs_does_not_push_corruption_below_zero() -> None:
    assert rhs(0.0, 0.0, RESISTER, constant_usage(1.0)) == 0.0
    assert rhs(0.0, -1e-9, RESISTER, constant_usage(1.0)) == 0.0


def test_rhs_rejects_negative_usage() -> None:
    with pytest.raises(ParameterError):
        rhs(0.0, 0.1, FALLER, lambda _t: -1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"W": -0.1, "A": 0.5},
        {"W": 0.5, "A": -0.1},
        {"W": 0.5, "A": 0.5, "alpha": 0.0},
        {"W": 0.5, "A": 0.5, "beta": -1.0},
        {"W": 0.5, "A": 0.5, "gamma": -1.0},
        {"W": 0.5, "A": 0.5, "c_star": 0.0},
        {"W": 0.5, "A": 0.5, "c_star": 1.5},
        {"W": 0.5, "A": 0.5, "c0": 0.95},
        {"W": math.nan, "A": 0.5},
        {"W": 0.5, "A": math.inf},
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ParameterError):
        BearerParams(**kwargs)


def test_constant_usage_rejects_negative_and_nan() -> None:
    with pytest.raises(ParameterError):
        constant_usage(-0.1)
    with pytest.raises(ParameterError):
        constant_usage(math.nan)


def test_fixed_point_balances_the_drift() -> None:
    c_u = unstable_fixed_point(RESISTER)
    assert 0 < c_u < 1
    assert rhs(0.0, c_u, RESISTER, constant_usage(1.0)) == pytest.approx(0.0, abs=1e-15)


def test_fixed_point_is_unstable() -> None:
    c_u = unstable_fixed_point(RESISTER)
    u = constant_usage(1.0)
    assert rhs(0.0, c_u + 1e-3, RESISTER, u) > 0
    assert rhs(0.0, c_u - 1e-3, RESISTER, u) < 0


def test_fixed_point_edge_cases_without_restoring_force() -> None:
    no_will = BearerParams(W=0.0, A=0.5, beta=0.0)
    assert unstable_fixed_point(no_will) == -math.inf
    inert = BearerParams(W=0.0, A=0.0)
    assert math.isnan(unstable_fixed_point(inert))
    assert fate(no_will) is Fate.FALLS
    assert fate(inert) is Fate.BALANCED


def test_critical_ambition_is_where_the_fixed_point_hits_zero() -> None:
    p = BearerParams(W=0.6, A=0.0)
    a_c = critical_ambition(p)
    at_critical = BearerParams(W=0.6, A=a_c)
    assert unstable_fixed_point(at_critical) == pytest.approx(0.0, abs=1e-12)


def test_critical_ambition_does_not_depend_on_beta() -> None:
    values = {critical_ambition(BearerParams(W=0.6, A=0.5, beta=b)) for b in (0.0, 1.0, 10.0)}
    assert len(values) == 1


def test_critical_ambition_without_usage_is_infinite() -> None:
    assert critical_ambition(FALLER, u=0.0) == math.inf


def test_fate_classification() -> None:
    assert fate(FALLER) is Fate.FALLS
    assert fate(RESISTER) is Fate.RESISTS
    c_u = unstable_fixed_point(RESISTER)
    balanced = BearerParams(W=0.9, A=0.3, c0=c_u)
    assert fate(balanced) is Fate.BALANCED


def test_no_usage_means_no_fall() -> None:
    assert fate(FALLER, u=0.0) is Fate.RESISTS
    assert analytic_fall_time(FALLER, u=0.0) == math.inf


def test_analytic_solution_starts_at_c0_and_reaches_c_star_at_fall_time() -> None:
    t_fall = analytic_fall_time(FALLER)
    c = analytic_solution(FALLER, np.array([0.0, t_fall]))
    assert c[0] == pytest.approx(FALLER.c0)
    assert c[1] == pytest.approx(FALLER.c_star)


def test_analytic_solution_drains_a_resister_to_zero() -> None:
    c = analytic_solution(RESISTER, np.array([0.0, 1e4]))
    assert c[-1] == 0.0


def test_analytic_fall_time_without_feedback_or_will_is_linear() -> None:
    p = BearerParams(W=0.0, A=0.5, beta=0.0)
    _, b = drift_coefficients(p)
    assert analytic_fall_time(p) == pytest.approx((p.c_star - p.c0) / b)


def test_simulate_detects_the_fall_and_stops_there() -> None:
    traj = simulate(FALLER, t_max=500.0)
    assert traj.fell
    assert traj.t[-1] == traj.fall_time
    assert traj.c[-1] == FALLER.c_star
    assert np.all(np.diff(traj.t) >= 0)


def test_simulate_resister_runs_to_the_horizon() -> None:
    traj = simulate(RESISTER, t_max=200.0)
    assert not traj.fell
    assert traj.t[-1] == pytest.approx(200.0)
    assert traj.c.min() >= 0.0


def test_simulate_accepts_time_varying_usage() -> None:
    def pulsed(t: float) -> float:
        return 1.0 if (t % 10.0) < 5.0 else 0.0

    steady = simulate(FALLER, t_max=500.0)
    intermittent = simulate(FALLER, t_max=500.0, usage=pulsed, max_step=0.5)
    assert intermittent.fell
    assert intermittent.fall_time > steady.fall_time


@pytest.mark.parametrize("t_max", [0.0, -1.0, math.inf])
def test_simulate_rejects_bad_horizon(t_max: float) -> None:
    with pytest.raises(ParameterError):
        simulate(FALLER, t_max=t_max)


def test_simulate_rejects_too_few_points() -> None:
    with pytest.raises(ParameterError):
        simulate(FALLER, t_max=10.0, n_points=1)


bearer = st.builds(
    BearerParams,
    W=st.floats(0.0, 1.0),
    A=st.floats(0.0, 1.0),
    beta=st.floats(0.0, 5.0),
    c0=st.floats(0.0, 0.5),
)


@settings(max_examples=60, deadline=None)
@given(p=bearer)
def test_corruption_stays_between_zero_and_threshold(p: BearerParams) -> None:
    traj = simulate(p, t_max=300.0, n_points=50)
    assert traj.c.min() >= 0.0
    assert traj.c.max() <= p.c_star + 1e-9


@settings(max_examples=60, deadline=None)
@given(p=bearer, extra=st.floats(0.01, 0.5))
def test_more_ambition_never_delays_the_fall(p: BearerParams, extra: float) -> None:
    hungrier = BearerParams(W=p.W, A=p.A + extra, beta=p.beta, c0=p.c0)
    assert analytic_fall_time(hungrier) <= analytic_fall_time(p)


@settings(max_examples=60, deadline=None)
@given(p=bearer, extra=st.floats(0.01, 0.5))
def test_more_will_never_hastens_the_fall(p: BearerParams, extra: float) -> None:
    stronger = BearerParams(W=p.W + extra, A=p.A, beta=p.beta, c0=p.c0)
    assert analytic_fall_time(stronger) >= analytic_fall_time(p)


def test_fall_time_stays_finite_when_will_is_subnormal() -> None:
    # Regression: with gamma*W a nonzero subnormal, C_u = -b/a overflowed to -inf and the old
    # formula gave nan. W must keep gamma*W above zero: 5e-324 underflows to a = 0 and takes the
    # linear branch, which never had the bug.
    p = BearerParams(W=2.2250738585e-313, A=0.5, beta=0.0)
    assert p.gamma * p.W > 0.0
    expected = BearerParams(W=0.0, A=0.5, beta=0.0)
    assert fate(p) is Fate.FALLS
    assert analytic_fall_time(p) == pytest.approx(analytic_fall_time(expected))


def test_fall_time_formula_matches_the_log_form() -> None:
    a, b = drift_coefficients(FALLER)
    c_u = -b / a
    log_form = math.log((FALLER.c_star - c_u) / (FALLER.c0 - c_u)) / a
    assert analytic_fall_time(FALLER) == pytest.approx(log_form, rel=1e-12)


def test_fall_time_stays_finite_for_a_subnormal_start_above_the_fixed_point() -> None:
    # Regression: C0 = 2e-309 just above C_u = 0 overflowed the linear time.
    p = BearerParams(W=0.5, A=0.5, beta=0.0, c0=2.225073858507203e-309)
    a, _ = drift_coefficients(p)
    assert analytic_fall_time(p) == pytest.approx((math.log(p.c_star) - math.log(p.c0)) / a)
