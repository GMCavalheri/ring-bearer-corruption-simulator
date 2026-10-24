"""Plotly figures for the Streamlit app: the animated fall and the bifurcation view."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from corruption.bifurcation import fixed_point
from corruption.model import BearerParams, FloatArray, Trajectory, critical_ambition
from corruption.stochastic import MonteCarloResult

FALL_COLOR = "#b3123c"
LINE_COLOR = "#6b4fbb"


def _resample(traj: Trajectory, n_frames: int) -> tuple[FloatArray, FloatArray]:
    t = np.linspace(0.0, float(traj.t[-1]), n_frames)
    return t, np.interp(t, traj.t, traj.c)


def fall_animation(traj: Trajectory, p: BearerParams, *, n_frames: int = 80) -> go.Figure:
    """C(t) drawn frame by frame, with a play button and the fall threshold."""
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2")
    t, c = _resample(traj, n_frames)
    t_end = float(t[-1]) if t[-1] > 0 else 1.0

    def frame_data(k: int) -> list[go.Scatter]:
        fallen = traj.fell and k == n_frames - 1
        return [
            go.Scatter(
                x=t[: k + 1],
                y=c[: k + 1],
                mode="lines",
                line={"color": LINE_COLOR, "width": 3},
                name="C(t)",
            ),
            go.Scatter(
                x=[t[k]],
                y=[c[k]],
                mode="markers+text",
                marker={
                    "size": 16 if fallen else 12,
                    "color": FALL_COLOR if fallen else LINE_COLOR,
                },
                text=["wraith" if fallen else ""],
                textposition="top center",
                name="bearer",
            ),
        ]

    fig = go.Figure(
        data=frame_data(0),
        frames=[go.Frame(data=frame_data(k), name=str(k)) for k in range(n_frames)],
    )
    fig.add_hline(
        y=p.c_star,
        line={"dash": "dash", "color": FALL_COLOR},
        annotation_text="C* — the fall",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis={"title": "years of ring use", "range": [0, t_end * 1.02]},
        yaxis={"title": "corruption C", "range": [0, 1]},
        showlegend=False,
        height=430,
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "direction": "left",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 40, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
                    },
                ],
            }
        ],
    )
    return fig


def bifurcation_view(p: BearerParams, u: float = 1.0, *, a_max: float = 1.5) -> go.Figure:
    """Point of no return C_u against A for this bearer's β, with the bearer marked."""
    a_c = critical_ambition(p, u)
    a_hi = max(a_max, p.A * 1.1)
    A = np.linspace(0.0, a_hi, 300)
    below = A[a_c >= A]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=below,
            y=fixed_point(p, below, p.beta, u),
            mode="lines",
            line={"dash": "dash", "color": FALL_COLOR, "width": 2},
            name="C_u (point of no return)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=below,
            y=np.zeros_like(below),
            mode="lines",
            line={"color": "black", "width": 4},
            name="C = 0 (resists)",
        )
    )
    falls = p.c0 > fixed_point(p, p.A, p.beta, u)
    fig.add_trace(
        go.Scatter(
            x=[p.A],
            y=[p.c0],
            mode="markers",
            marker={"size": 14, "symbol": "star", "color": FALL_COLOR if falls else "#1b7f3b"},
            name="this bearer (A, C0)",
        )
    )
    if math.isfinite(a_c):
        fig.add_vline(x=a_c, line={"color": "gray"}, annotation_text=f"A_c = {a_c:.2f}")
    fig.update_layout(
        xaxis={"title": "ambition A", "range": [0, a_hi]},
        yaxis={"title": "corruption", "range": [-0.02, 1]},
        height=430,
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def fall_time_histogram(result: MonteCarloResult) -> go.Figure:
    """Histogram of Monte Carlo fall times."""
    fell = result.fall_time[result.fell]
    fig = go.Figure(go.Histogram(x=fell, nbinsx=50, marker={"color": FALL_COLOR}))
    fig.update_layout(
        xaxis={"title": "years until the fall"},
        yaxis={"title": "bearers"},
        height=380,
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
        title=f"{result.fraction_fallen:.1%} fell within {result.horizon:g} years",
    )
    return fig
