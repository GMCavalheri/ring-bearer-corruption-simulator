"""Matplotlib figure builders shared by the notebooks and the README figures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from corruption.model import Trajectory


def plot_trajectories(
    trajectories: Sequence[Trajectory],
    labels: Sequence[str],
    c_star: float,
    *,
    title: str = "Corruption of the bearers",
) -> Figure:
    """C(t) for several bearers, with the fall threshold and a marker at each fall."""
    if len(trajectories) != len(labels):
        raise ValueError("trajectories and labels must have the same length")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for traj, label in zip(trajectories, labels, strict=True):
        (line,) = ax.plot(traj.t, traj.c, label=label, lw=1.8)
        if traj.fell:
            ax.plot(traj.fall_time, c_star, "o", color=line.get_color(), ms=5)
    ax.axhline(c_star, color="0.3", ls="--", lw=1, label="C* (fall)")
    ax.set(xlabel="years of ring use", ylabel="corruption C", ylim=(0, 1), title=title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.grid(alpha=0.3)
    return fig
