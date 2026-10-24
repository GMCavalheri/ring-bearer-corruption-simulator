import math

import numpy as np
import pytest

from corruption.bifurcation import (
    critical_ambition,
    fate_boundary_ambition,
    fixed_point,
    sweep,
)
from corruption.model import BearerParams, ParameterError

BASE = BearerParams(W=0.6, A=0.5, c0=0.05)
A_AXIS = np.linspace(0.0, 1.5, 61)
BETA_AXIS = np.array([0.0, 1.0, 3.0, 8.0])


def test_sweep_shapes_follow_beta_rows_and_ambition_columns() -> None:
    result = sweep(BASE, A_AXIS, BETA_AXIS)
    assert result.fixed_point.shape == (4, 61)
    assert result.falls.shape == (4, 61)
    assert result.fall_time.shape == (4, 61)


def test_fixed_point_crosses_zero_at_the_critical_ambition_for_every_beta() -> None:
    a_c = critical_ambition(BASE)
    for beta in BETA_AXIS:
        assert fixed_point(BASE, a_c, beta) == pytest.approx(0.0, abs=1e-12)
        assert fixed_point(BASE, a_c * 0.99, beta) > 0
        assert fixed_point(BASE, a_c * 1.01, beta) < 0


def test_fixed_point_without_ambition_is_one() -> None:
    assert fixed_point(BASE, 0.0, 3.0) == pytest.approx(1.0)


def test_fixed_point_without_restoring_force() -> None:
    no_will = BearerParams(W=0.0, A=0.5)
    assert fixed_point(no_will, 0.5, 0.0) == -math.inf
    assert math.isnan(fixed_point(no_will, 0.0, 0.0))


def test_falls_matches_the_fate_boundary() -> None:
    result = sweep(BASE, A_AXIS, BETA_AXIS)
    boundary = fate_boundary_ambition(BASE, BETA_AXIS)
    expected = A_AXIS[None, :] > boundary[:, None]
    np.testing.assert_array_equal(result.falls, expected)


def test_fate_boundary_equals_critical_ambition_for_a_clean_start() -> None:
    clean = BearerParams(W=0.6, A=0.5, c0=0.0)
    np.testing.assert_allclose(fate_boundary_ambition(clean, BETA_AXIS), critical_ambition(clean))


def test_feedback_lowers_the_fate_boundary_for_a_corrupted_start() -> None:
    boundary = fate_boundary_ambition(BASE, BETA_AXIS)
    assert np.all(np.diff(boundary) < 0)


def test_fall_time_is_infinite_exactly_where_the_bearer_resists() -> None:
    result = sweep(BASE, A_AXIS, BETA_AXIS)
    np.testing.assert_array_equal(np.isinf(result.fall_time), ~result.falls)


def test_fall_time_shrinks_with_ambition_and_feedback() -> None:
    result = sweep(BASE, A_AXIS, BETA_AXIS)
    t = np.where(result.falls, result.fall_time, np.nan)
    rows_ok = [np.all(np.diff(row[~np.isnan(row)]) < 0) for row in t]
    assert all(rows_ok)
    last_col = t[:, -1]
    assert np.all(np.diff(last_col) < 0)


def test_sweep_without_will_or_feedback_uses_linear_fall_time() -> None:
    no_will = BearerParams(W=0.0, A=0.5, c0=0.05)
    result = sweep(no_will, [0.5], [0.0])
    expected = (no_will.c_star - no_will.c0) / (no_will.alpha * 0.5)
    assert result.fall_time[0, 0] == pytest.approx(expected)


def test_no_usage_means_nobody_falls() -> None:
    result = sweep(BASE, A_AXIS, BETA_AXIS, u=0.0)
    assert not result.falls.any()
    assert np.all(np.isinf(fate_boundary_ambition(BASE, BETA_AXIS, u=0.0)))


@pytest.mark.parametrize(
    ("A_values", "beta_values"),
    [([], [1.0]), ([0.5], []), ([[0.5]], [1.0]), ([-0.1], [1.0]), ([0.5], [math.nan])],
)
def test_sweep_rejects_bad_axes(A_values: list[float], beta_values: list[float]) -> None:
    with pytest.raises(ParameterError):
        sweep(BASE, A_values, beta_values)


def test_sweep_rejects_negative_usage() -> None:
    with pytest.raises(ParameterError):
        sweep(BASE, A_AXIS, BETA_AXIS, u=-1.0)
