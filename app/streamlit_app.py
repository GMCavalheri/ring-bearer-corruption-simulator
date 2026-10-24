"""Ring-Bearer Corruption Simulator: interactive explorer.

Run with `uv run streamlit run app/streamlit_app.py`.
"""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from corruption.flags import is_enabled
from corruption.interactive import bifurcation_view, fall_animation, fall_time_histogram
from corruption.model import (
    BearerParams,
    Fate,
    critical_ambition,
    fate,
    simulate,
    unstable_fixed_point,
)
from corruption.stochastic import Population, SDEConfig, monte_carlo, sample_population

st.set_page_config(page_title="Ring-Bearer Corruption Simulator", page_icon="💍", layout="wide")

st.title("Ring-Bearer Corruption Simulator")
st.caption(
    "How long does a bearer resist before becoming a wraith? "
    "dC/dt = α·u·A·(1 + β·C) − γ·W·(1 − C). All parameter values are invented."
)

with st.sidebar:
    st.header("The bearer")
    W = st.slider("Will W", 0.0, 1.0, 0.60, 0.01, help="Strength that drains corruption.")
    A = st.slider("Ambition A", 0.0, 1.0, 0.75, 0.01, help="How strongly ring use feeds it.")
    beta = st.slider(
        "Feedback β",
        0.0,
        10.0,
        3.0,
        0.1,
        help="How much existing corruption accelerates its own growth.",
    )
    with st.expander("Advanced"):
        alpha = st.slider("Corruption rate α", 0.005, 0.1, 0.02, 0.005)
        gamma = st.slider("Recovery rate γ", 0.0, 0.1, 0.02, 0.005)
        c_star = st.slider("Fall threshold C*", 0.5, 1.0, 0.9, 0.01)
        c0 = st.slider("Initial corruption C0", 0.0, 0.45, 0.05, 0.01)
        u = st.slider("Ring usage u", 0.0, 1.0, 1.0, 0.05, help="Fraction of time worn.")
        horizon = st.slider("Horizon (years)", 20, 500, 200, 10)

p = BearerParams(W=W, A=A, alpha=alpha, beta=beta, gamma=gamma, c_star=c_star, c0=c0)
outcome = fate(p, u)
traj = simulate(p, t_max=float(horizon), usage=lambda _t: u, n_points=400)
c_u = unstable_fixed_point(p, u)
a_c = critical_ambition(p, u)

col_fate, col_time, col_cu, col_ac = st.columns(4)
if traj.fell:
    col_fate.metric("Fate", "Falls")
    col_time.metric("Becomes a wraith after", f"{traj.fall_time:.1f} years")
elif outcome is Fate.FALLS:
    col_fate.metric("Fate", "Falls (later)")
    col_time.metric("Becomes a wraith after", f"> {horizon} years")
else:
    col_fate.metric("Fate", "Resists")
    col_time.metric("Becomes a wraith after", "never")
col_cu.metric("Point of no return C_u", f"{c_u:.3f}" if math.isfinite(c_u) else "none")
col_ac.metric("Critical ambition A_c", f"{a_c:.3f}" if math.isfinite(a_c) else "∞")

tab_names = ["Fall animation", "Bifurcation"]
sde_enabled = is_enabled("EXPERIMENTAL_SDE")
if sde_enabled:
    tab_names.append("Stochastic (experimental)")
tabs = st.tabs(tab_names)

with tabs[0]:
    st.plotly_chart(fall_animation(traj, p), width="stretch")
    st.markdown(
        "Press **Play** to watch corruption grow. "
        f"The bearer starts at C0 = {c0:.2f}; if that is above the point of no return "
        "C_u, corruption feeds on itself until it reaches C*."
    )

with tabs[1]:
    st.plotly_chart(bifurcation_view(p, u), width="stretch")
    st.markdown(
        "Below the dashed curve corruption drains away; above it the bearer falls. "
        "The curve reaches zero at A_c = γW/(αu) for every β: feedback changes how much "
        "initial corruption a bearer can survive, not the ambition at which nobody can."
    )

if sde_enabled:
    with tabs[2]:
        st.markdown(
            "Bearers drawn around this bearer's W, A and C0 (Beta distributions), "
            "with noise σ dB added to the ODE and solved by Euler–Maruyama."
        )
        sigma = st.slider("Noise σ", 0.0, 0.1, 0.02, 0.005)
        n = st.select_slider("Bearers", options=[100, 250, 500, 1000, 2000], value=500)
        seed = st.number_input("Seed", min_value=0, value=2026, step=1)
        clipped = [min(max(x, 0.01), 0.99) for x in (W, A, max(c0, 0.01))]
        pop = sample_population(
            int(n),
            np.random.default_rng(int(seed)),
            W_mean=clipped[0],
            A_mean=clipped[1],
            c0_mean=clipped[2],
        )
        pop = Population(W=pop.W, A=pop.A, c0=np.minimum(pop.c0, c_star * 0.99))
        cfg = SDEConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            c_star=c_star,
            sigma=sigma,
            u=u,
            dt=0.1,
            t_max=float(horizon),
        )
        result = monte_carlo(pop, cfg, seed=int(seed))
        st.plotly_chart(fall_time_histogram(result), width="stretch")
else:
    st.caption("Experimental stochastic tab is off. Set `RBCS_EXPERIMENTAL_SDE=1` to enable it.")
