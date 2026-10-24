"""Matplotlib figure builders shared by the notebooks and the README figures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from corruption.bifurcation import Sweep, fate_boundary_ambition, fixed_point
from corruption.model import BearerParams, FloatArray, Trajectory, critical_ambition
from corruption.stochastic import MonteCarloResult


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


def plot_bifurcation(
    base: BearerParams,
    A_values: FloatArray,
    betas: Sequence[float],
    u: float = 1.0,
) -> Figure:
    """Equilibria against ambition: stable C = 0 and the unstable point of no return."""
    a_c = critical_ambition(base, u)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    below = A_values[A_values <= a_c]
    above = A_values[A_values >= a_c]
    ax.plot(below, np.zeros_like(below), color="black", lw=3, label="C = 0 (stable: resists)")
    cmap = plt.get_cmap("viridis")
    for i, beta in enumerate(betas):
        c_u = fixed_point(base, below, beta, u)
        ax.plot(
            below,
            c_u,
            "--",
            color=cmap(i / max(len(betas) - 1, 1)),
            lw=2,
            label=f"C_u, β = {beta:g} (unstable)",
        )
    ax.plot(above, np.zeros_like(above), ":", color="black", lw=1.5)
    ax.axvline(a_c, color="crimson", lw=1, label=f"A_c = γW/(αu) = {a_c:.2f}")
    ax.axhline(base.c0, color="0.4", lw=0.8, ls="-.", label=f"C0 = {base.c0:g}")
    ax.text(a_c * 0.45, 0.82, "above C_u: falls", color="crimson")
    ax.text(a_c * 1.02, 0.5, "no resisting state:\nevery bearer falls", color="crimson")
    ax.set(
        xlabel="ambition A",
        ylabel="equilibrium corruption",
        title=f"Who resists forever and who falls (W = {base.W:g})",
        xlim=(A_values[0], A_values[-1]),
        ylim=(-0.03, 1),
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    return fig


def plot_fate_map(result: Sweep, base: BearerParams, u: float = 1.0) -> Figure:
    """Fall time over (A, β); grey where the bearer resists, with both boundaries."""
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    finite = np.where(np.isfinite(result.fall_time), result.fall_time, np.nan)
    mesh = ax.pcolormesh(
        result.A, result.beta, finite, shading="auto", cmap="magma_r", norm=LogNorm()
    )
    ax.set_facecolor("0.85")
    fig.colorbar(mesh, ax=ax, label="years until the fall")
    ax.plot(
        fate_boundary_ambition(base, result.beta, u),
        result.beta,
        color="white",
        lw=2,
        label="fate boundary A_f(β)",
    )
    ax.axvline(critical_ambition(base, u), color="cyan", lw=1.5, ls="--", label="A_c")
    ax.text(0.02, 0.5, "resists\nforever", transform=ax.transAxes, color="0.3")
    ax.set(
        xlabel="ambition A",
        ylabel="feedback β",
        xlim=(result.A[0], result.A[-1]),
        title=f"Time to fall (W = {base.W:g}, C0 = {base.c0:g})",
    )
    ax.legend(loc="upper right", fontsize=9)
    return fig


def plot_sample_paths(
    t: FloatArray, paths: FloatArray, c_star: float, *, title: str = "Sample paths"
) -> Figure:
    """Stochastic C(t) paths, one thin line per bearer."""
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(t, paths, lw=0.8, alpha=0.7)
    ax.axhline(c_star, color="0.3", ls="--", lw=1, label="C* (fall)")
    ax.set(xlabel="years of ring use", ylabel="corruption C", ylim=(0, 1), title=title)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    return fig


def plot_fall_time_distribution(
    result: MonteCarloResult, *, title: str = "Time to fall", bins: int = 60
) -> Figure:
    """Histogram of fall times next to the empirical CDF, with censoring noted."""
    fell = result.fall_time[result.fell]
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ax_hist.hist(fell, bins=bins, color="#7a1f3d", alpha=0.85)
    if fell.size:
        ax_hist.axvline(
            float(np.median(fell)),
            color="black",
            ls="--",
            lw=1,
            label=f"median {np.median(fell):.1f} y",
        )
        ax_hist.legend(frameon=False)
    ax_hist.set(xlabel="years until the fall", ylabel="bearers", title=title)
    ordered = np.sort(fell)
    ecdf = np.arange(1, ordered.size + 1) / len(result.fall_time)
    ax_cdf.step(ordered, ecdf, where="post", color="#7a1f3d")
    ax_cdf.axhline(
        result.fraction_fallen,
        color="0.4",
        ls=":",
        lw=1,
        label=f"{result.fraction_fallen:.1%} fallen by {result.horizon:g} y",
    )
    ax_cdf.set(
        xlabel="years",
        ylabel="fraction fallen",
        ylim=(0, 1.02),
        title="Empirical CDF (censored at the horizon)",
    )
    ax_cdf.legend(loc="lower right", frameon=False)
    for ax in (ax_hist, ax_cdf):
        ax.grid(alpha=0.3)
    return fig
