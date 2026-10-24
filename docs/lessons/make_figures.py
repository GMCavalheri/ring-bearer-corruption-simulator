"""Teaching figures for the lesson series (docs/figures/lessons/).

Run with ``uv run python docs/lessons/make_figures.py``. The notebooks make the
results figures in ``docs/figures/``; these illustrate ideas. Every figure is
computed from the ``corruption`` package with fixed seeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

from corruption.model import (
    BearerParams,
    analytic_fall_time,
    critical_ambition,
    drift_coefficients,
    simulate,
    unstable_fixed_point,
)
from corruption.races import survival_table
from corruption.stochastic import Population, SDEConfig, monte_carlo, sample_population

OUT = Path(__file__).resolve().parents[1] / "figures" / "lessons"
FALL = "#b3123c"
RESIST = "#1b7f3b"
NEUTRAL = "#6b4fbb"
SEED = 20261023


def save(fig: Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- Lesson 1


def l1_phase_line() -> None:
    """Phase line and trajectories on both sides of the point of no return."""
    p = BearerParams(W=0.8, A=0.4)
    c_u = unstable_fixed_point(p)
    fig, (ax_pl, ax_tr) = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")

    a, b = drift_coefficients(p)
    c = np.linspace(0.0, 1.0, 201)
    ax_pl.plot(c, a * c + b, color=NEUTRAL, lw=2, label="dC/dt = aC + b")
    ax_pl.axhline(0, color="black", lw=0.8)
    ax_pl.plot([c_u], [0], "o", ms=9, mfc="white", mec=FALL, mew=2, label=f"C_u = {c_u:.2f}")
    ax_pl.plot([0], [0], "o", ms=9, color=RESIST, label="C = 0 (held by the clamp)")
    for x0, x1 in [(0.15, 0.05), (0.3, 0.45), (0.65, 0.8)]:
        ax_pl.annotate(
            "",
            xy=(x1, 0),
            xytext=(x0, 0),
            arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 1.5},
        )
    ax_pl.set(
        xlabel="corruption C", ylabel="dC/dt  [1/year]", title="Phase line (W = 0.8, A = 0.4)"
    )
    ax_pl.legend(frameon=False, loc="upper left")
    ax_pl.grid(alpha=0.3)

    t = np.linspace(0, 120, 400)
    for c0 in (0.05, 0.12, 0.18, 0.22, 0.28, 0.4):
        q = BearerParams(W=p.W, A=p.A, c0=c0)
        traj = simulate(q, t_max=120.0)
        color = FALL if c0 > c_u else RESIST
        ax_tr.plot(traj.t, traj.c, color=color, lw=1.8)
        ax_tr.text(-3, c0, f"{c0:.2f}", ha="right", va="center", fontsize=8, color=color)
    ax_tr.plot(t, np.full_like(t, c_u), ":", color="0.4", label="C_u, the point of no return")
    ax_tr.axhline(p.c_star, color="0.3", ls="--", lw=1, label="C* (fall)")
    ax_tr.set(
        xlabel="years of ring use",
        ylabel="corruption C",
        ylim=(0, 1),
        xlim=(-12, 120),
        title="Same character, different starting corruption C0",
    )
    ax_tr.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax_tr.grid(alpha=0.3)
    save(fig, "l1_phase_line")


def l1_solver_tolerance() -> None:
    """Fall-time error of solve_ivp against the closed form, as the tolerance tightens."""
    p = BearerParams(W=0.60, A=0.78)
    exact = analytic_fall_time(p)
    rtols = np.logspace(-3, -11, 17)
    errors = [abs(simulate(p, 100.0, rtol=r, atol=r * 1e-2).fall_time - exact) for r in rtols]
    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    ax.loglog(rtols, errors, "o-", color=NEUTRAL)
    ax.set(
        xlabel="relative tolerance rtol (atol = rtol / 100)",
        ylabel="|t_fall numeric − t_fall exact|  [years]",
        title="RK45 fall time against the closed form (bearer 5)",
    )
    ax.invert_xaxis()
    ax.grid(alpha=0.3, which="both")
    save(fig, "l1_solver_tolerance")


# ---------------------------------------------------------------- Lesson 2


def l2_phase_lines() -> None:
    """Phase lines below, at and above the critical ambition."""
    base = BearerParams(W=0.6, A=0.0)
    a_c = critical_ambition(base)
    c = np.linspace(0.0, 1.0, 201)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, layout="constrained")
    for ax, factor in zip(axes, (0.6, 1.0, 1.4), strict=True):
        p = BearerParams(W=base.W, A=a_c * factor)
        a, b = drift_coefficients(p)
        ax.plot(c, a * c + b, color=NEUTRAL, lw=2)
        ax.axhline(0, color="black", lw=0.8)
        c_u = unstable_fixed_point(p)
        if factor < 1:
            ax.plot([0], [0], "o", ms=9, color=RESIST)
            ax.plot([c_u], [0], "o", ms=9, mfc="white", mec=FALL, mew=2)
            ax.text(c_u + 0.03, -0.006, f"C_u = {c_u:.2f}", color=FALL)
            ax.text(0.35, 0.045, "resist below C_u,\nfall above it", fontsize=9)
        elif factor == 1:
            ax.plot([0], [0], "o", ms=9, mfc="white", mec=FALL, mew=2)
            ax.text(0.05, 0.045, "C_u has reached 0:\nthe resisting state is gone", fontsize=9)
        else:
            ax.text(
                0.05, 0.045, "drift > 0 everywhere:\nno equilibrium, everyone falls", fontsize=9
            )
        ax.set(xlabel="corruption C", title=f"A = {factor:g} A_c  ({p.A:.2f})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("dC/dt  [1/year]")
    fig.suptitle("Filled: stable · open: unstable.  W = 0.6, β = 3")
    save(fig, "l2_phase_lines")


def l2_fate_plane() -> None:
    """Falls/resists regions in the (A, C0) plane for three values of β."""
    base = BearerParams(W=0.6, A=0.0)
    a_c = critical_ambition(base)
    A = np.linspace(0.0, 1.0, 400)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    colors = {0.0: "#2b6cb0", 3.0: NEUTRAL, 8.0: FALL}
    for beta, color in colors.items():
        a = base.alpha * A * beta + base.gamma * base.W
        b = base.alpha * A - base.gamma * base.W
        c_u = np.clip(-b / a, 0, 1)
        ax.plot(A, c_u, color=color, lw=2, label=f"C_u(A), β = {beta:g}")
        if beta == 3.0:
            ax.fill_between(A, c_u, 1, color=FALL, alpha=0.08)
            ax.fill_between(A, 0, c_u, color=RESIST, alpha=0.08)
    ax.axvline(a_c, color="0.3", ls="--", lw=1)
    ax.text(a_c + 0.01, 0.92, f"A_c = {a_c:.2f}\n(every β)", fontsize=9)
    ax.text(0.7, 0.45, "falls\n(shaded for β = 3)", color=FALL)
    ax.text(0.04, 0.06, "resists", color=RESIST)
    ax.set(
        xlabel="ambition A",
        ylabel="initial corruption C0",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Fate in the (A, C0) plane, W = 0.6",
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.3)
    save(fig, "l2_fate_plane")


# ---------------------------------------------------------------- Lesson 3


def l3_wiener_paths() -> None:
    """Brownian motion: independent Gaussian steps, spread growing like √t."""
    rng = np.random.default_rng(SEED)
    dt, n = 0.05, 2000
    t = np.arange(n + 1) * dt
    steps = rng.standard_normal((n, 8)) * np.sqrt(dt)
    paths = np.vstack([np.zeros(8), np.cumsum(steps, axis=0)])
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    ax.plot(t, paths, lw=0.9, alpha=0.8)
    ax.plot(t, 2 * np.sqrt(t), "k--", lw=1, label="±2√t (95% band at each t)")
    ax.plot(t, -2 * np.sqrt(t), "k--", lw=1)
    ax.set(xlabel="t", ylabel="B(t)", title="Eight Wiener paths")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    save(fig, "l3_wiener_paths")


def l3_em_convergence() -> None:
    """Euler–Maruyama without noise is Euler's method: error proportional to dt."""
    p = BearerParams(W=0.5, A=0.8)
    exact = analytic_fall_time(p)
    one = Population(W=np.array([p.W]), A=np.array([p.A]), c0=np.array([p.c0]))
    dts = np.array([0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125])
    errors = [
        abs(monte_carlo(one, SDEConfig(sigma=0.0, dt=float(dt)), seed=0).fall_time[0] - exact)
        for dt in dts
    ]
    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    ax.loglog(dts, errors, "o-", color=NEUTRAL, label="Euler–Maruyama, σ = 0")
    ax.loglog(dts, errors[-1] * dts / dts[-1], "k--", lw=1, label="slope 1 (first order)")
    ax.set(
        xlabel="time step dt [years]",
        ylabel="fall-time error [years]",
        title="Convergence of the fall time (W = 0.5, A = 0.8)",
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    save(fig, "l3_em_convergence")


def l3_mc_error() -> None:
    """A Monte Carlo estimate and its 95% band as the number of bearers grows."""
    pop = sample_population(10_000, np.random.default_rng(SEED))
    result = monte_carlo(pop, SDEConfig(), seed=SEED)
    # Censored bearers count at the horizon, so every sample is finite.
    observed = result.observed_time()
    n = np.arange(1, len(observed) + 1)
    running = np.cumsum(observed) / n
    running_sq = np.cumsum(observed**2) / n
    se = np.sqrt(np.maximum(running_sq - running**2, 0) / n)
    fig, (ax_est, ax_se) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    ax_est.plot(n, running, color=NEUTRAL, lw=1.2, label="running mean of min(T, 200 y)")
    ax_est.fill_between(
        n,
        running - 1.96 * se,
        running + 1.96 * se,
        color=NEUTRAL,
        alpha=0.2,
        label="±1.96 standard errors",
    )
    ax_est.set(
        xscale="log",
        xlabel="bearers simulated N",
        ylabel="years",
        title="The estimate settles as N grows",
    )
    ax_est.legend(frameon=False)
    ax_est.grid(alpha=0.3, which="both")
    ax_se.loglog(n[9:], se[9:], color=NEUTRAL, label="standard error")
    ax_se.loglog(n[9:], se[-1] * np.sqrt(n[-1] / n[9:]), "k--", lw=1, label="∝ 1/√N")
    ax_se.set(
        xlabel="bearers simulated N",
        ylabel="standard error [years]",
        title="Ten times more bearers, √10 less error",
    )
    ax_se.legend(frameon=False)
    ax_se.grid(alpha=0.3, which="both")
    save(fig, "l3_mc_error")


def l3_noise_escape() -> None:
    """Noise pushes deterministic resisters past their point of no return."""
    n = 2000
    sigmas = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for A, color in ((0.3, RESIST), (0.4, NEUTRAL), (0.5, FALL)):
        p = BearerParams(W=0.9, A=A, c0=0.0)
        pop = Population(W=np.full(n, p.W), A=np.full(n, A), c0=np.zeros(n))
        fallen = [
            monte_carlo(pop, SDEConfig(sigma=float(s), t_max=300.0), seed=SEED).fraction_fallen
            for s in sigmas
        ]
        ax.plot(
            sigmas,
            fallen,
            "o-",
            color=color,
            label=f"A = {A}  (C_u = {unstable_fixed_point(p):.2f})",
        )
    ax.set(
        xlabel="noise σ [1/√year]",
        ylabel="fraction fallen within 300 years",
        ylim=(-0.02, 1.02),
        title="Bearers who would resist forever (W = 0.9, C0 = 0)",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "l3_noise_escape")


# ---------------------------------------------------------------- Lesson 4

KM_EXAMPLE = [(12.0, True), (20.0, True), (25.0, False), (30.0, True), (40.0, False), (45.0, True)]


def l4_km_by_hand() -> None:
    """Kaplan–Meier on six bearers, two of them censored."""
    times = [t for t, _ in KM_EXAMPLE]
    events = [e for _, e in KM_EXAMPLE]
    kmf = KaplanMeierFitter(label="Kaplan–Meier estimate").fit(times, events)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        color=NEUTRAL,
        lw=2.2,
        show_censors=True,
        censor_styles={"ms": 12, "marker": "|"},
    )
    for t, e in KM_EXAMPLE:
        s = float(kmf.survival_function_at_times(t).iloc[0])
        ax.annotate(
            "fall" if e else "censored",
            (t, s),
            textcoords="offset points",
            xytext=(4, 8),
            fontsize=8,
            color=FALL if e else "0.4",
        )
    ax.set(
        xlabel="years of ring use",
        ylabel="S(t)",
        ylim=(-0.03, 1.05),
        xlim=(0, 50),
        title="Six bearers: falls at 12, 20, 30, 45; censored at 25 and 40",
    )
    ax.grid(alpha=0.3)
    save(fig, "l4_km_by_hand")


def l4_cumulative_hazard() -> None:
    """Nelson–Aalen cumulative hazard by race: the slope is the hazard rate."""
    table = survival_table(n_per_race=3000, base=SDEConfig(), seed=SEED)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for race, group in table.groupby("race", observed=True, sort=False):
        naf = NelsonAalenFitter(label=str(race)).fit(group["time"], group["fell"])
        naf.plot_cumulative_hazard(ax=ax, ci_show=False, lw=2)
    ax.set(
        yscale="log",
        ylim=(1e-3, 10),
        xlabel="years of ring use",
        ylabel="cumulative hazard H(t)",
        title="How fast each race accumulates risk",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    save(fig, "l4_cumulative_hazard")


def main() -> None:
    l1_phase_line()
    l1_solver_tolerance()
    l2_phase_lines()
    l2_fate_plane()
    l3_wiener_paths()
    l3_em_convergence()
    l3_mc_error()
    l3_noise_escape()
    l4_km_by_hand()
    l4_cumulative_hazard()


if __name__ == "__main__":
    main()
