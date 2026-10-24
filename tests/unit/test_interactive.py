import numpy as np
import pytest

from corruption.interactive import bifurcation_view, fall_animation, fall_time_histogram
from corruption.model import BearerParams, critical_ambition, simulate
from corruption.stochastic import MonteCarloResult

FALLER = BearerParams(W=0.5, A=0.8)
RESISTER = BearerParams(W=0.9, A=0.3)


def test_fall_animation_has_one_frame_per_step_and_ends_at_the_fall() -> None:
    traj = simulate(FALLER, t_max=200.0)
    fig = fall_animation(traj, FALLER, n_frames=30)
    assert len(fig.frames) == 30
    last_line, last_marker = fig.frames[-1].data
    assert last_line.y[-1] == pytest.approx(FALLER.c_star)
    assert last_marker.text == ("wraith",)
    assert fig.layout.updatemenus[0].buttons[0].label == "▶ Play"


def test_fall_animation_for_a_resister_never_shows_a_wraith() -> None:
    traj = simulate(RESISTER, t_max=100.0)
    fig = fall_animation(traj, RESISTER, n_frames=10)
    assert all(frame.data[1].text == ("",) for frame in fig.frames)


def test_fall_animation_rejects_too_few_frames() -> None:
    with pytest.raises(ValueError):
        fall_animation(simulate(FALLER, t_max=10.0), FALLER, n_frames=1)


def test_bifurcation_view_marks_bearer_and_critical_ambition() -> None:
    fig = bifurcation_view(RESISTER)
    names = [trace.name for trace in fig.data]
    assert "this bearer (A, C0)" in names
    star = fig.data[names.index("this bearer (A, C0)")]
    assert star.marker.color == "#1b7f3b"
    assert any(f"{critical_ambition(RESISTER):.2f}" in a.text for a in fig.layout.annotations)


def test_bifurcation_view_without_usage_has_no_critical_line() -> None:
    fig = bifurcation_view(FALLER, u=0.0)
    assert not fig.layout.shapes


def test_fall_time_histogram_reports_the_fraction_fallen() -> None:
    result = MonteCarloResult(fall_time=np.array([10.0, 20.0, np.inf, np.inf]), horizon=50.0)
    fig = fall_time_histogram(result)
    assert "50.0% fell within 50 years" in fig.layout.title.text
