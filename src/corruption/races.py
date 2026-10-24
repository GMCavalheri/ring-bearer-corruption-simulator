"""Per-race parameter sets and survival analysis of the fall.

Each race draws bearers from its own Beta distributions of will, ambition and
initial corruption, and has its own recovery rate γ. Dwarves add a side effect:
a share κ of the ring's pull feeds greed G instead of submission,

    dG/dt = κ·α·u·A,   and corruption sees only (1 − κ)·α·u·A,

so their rings amplify wealth rather than obedience. Fall times from the
stochastic model become a survival table (time, event), and Kaplan–Meier curves
compare the races. Bearers still standing at the horizon are right-censored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from corruption.model import ParameterError
from corruption.stochastic import SDEConfig, monte_carlo, sample_population


@dataclass(frozen=True)
class Race:
    """Distribution of bearers for one race. All values are invented assumptions."""

    name: str
    W_mean: float
    A_mean: float
    c0_mean: float = 0.05
    concentration: float = 20.0
    gamma: float = 0.02
    greed_share: float = 0.0
    description: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        for attr in ("W_mean", "A_mean", "c0_mean"):
            if not 0 < getattr(self, attr) < 1:
                raise ParameterError(f"{attr} must be in (0, 1)")
        if self.concentration <= 0 or self.gamma < 0:
            raise ParameterError("concentration must be positive and gamma non-negative")
        if not 0 <= self.greed_share < 1:
            raise ParameterError("greed_share must be in [0, 1)")

    def sde_config(self, base: SDEConfig) -> SDEConfig:
        """The shared SDE settings with this race's γ and the greed diversion applied."""
        return replace(base, gamma=self.gamma, alpha=base.alpha * (1 - self.greed_share))


RACES: dict[str, Race] = {
    "Men": Race(
        name="Men",
        W_mean=0.55,
        A_mean=0.80,
        gamma=0.02,
        description="High ambition; the Nine were given to Men.",
    ),
    "Dwarves": Race(
        name="Dwarves",
        W_mean=0.60,
        A_mean=0.70,
        gamma=0.025,
        greed_share=0.5,
        description="Tough (high γ); the ring feeds greed for gold instead of submission.",
    ),
    "Elves": Race(
        name="Elves",
        W_mean=0.85,
        A_mean=0.45,
        c0_mean=0.02,
        gamma=0.03,
        description="Control group with strong will and little ambition.",
    ),
}

SURVIVAL_COLUMNS = ["race", "W", "A", "c0", "time", "fell", "greed"]


def survival_table(
    races: Mapping[str, Race] | None = None,
    *,
    n_per_race: int = 2000,
    base: SDEConfig | None = None,
    seed: int = 0,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Simulate every race and return one row per bearer.

    Columns: race, W, A, c0, time (fall time or the horizon), fell (event
    observed), greed (G at `time`).
    """
    races = RACES if races is None else races
    base = base if base is not None else SDEConfig()
    if not races:
        raise ParameterError("need at least one race")
    if n_per_race <= 0:
        raise ParameterError("n_per_race must be positive")
    frames = []
    children = np.random.SeedSequence(seed).spawn(len(races))
    for race, child in zip(races.values(), children, strict=True):
        pop_seed, mc_seed = child.spawn(2)
        pop = sample_population(
            n_per_race,
            np.random.default_rng(pop_seed),
            W_mean=race.W_mean,
            A_mean=race.A_mean,
            c0_mean=race.c0_mean,
            concentration=race.concentration,
        )
        result = monte_carlo(pop, race.sde_config(base), seed=mc_seed, n_workers=n_workers)
        time = result.observed_time()
        greed = race.greed_share * base.alpha * base.u * pop.A * time
        frames.append(
            pd.DataFrame(
                {
                    "race": race.name,
                    "W": pop.W,
                    "A": pop.A,
                    "c0": pop.c0,
                    "time": time,
                    "fell": result.fell,
                    "greed": greed,
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    table["race"] = pd.Categorical(table["race"], categories=[r.name for r in races.values()])
    return table


def kaplan_meier(table: pd.DataFrame) -> dict[str, KaplanMeierFitter]:
    """One fitted Kaplan–Meier estimator per race, in table order."""
    fitters: dict[str, KaplanMeierFitter] = {}
    for race, group in table.groupby("race", observed=True, sort=False):
        kmf = KaplanMeierFitter(label=str(race))
        kmf.fit(group["time"], event_observed=group["fell"])
        fitters[str(race)] = kmf
    return fitters


def median_fall_times(fitters: Mapping[str, KaplanMeierFitter]) -> pd.Series:
    """Median survival time per race (`inf` when fewer than half fall)."""
    return pd.Series(
        {name: float(kmf.median_survival_time_) for name, kmf in fitters.items()},
        name="median fall time",
    )


def pairwise_logrank(table: pd.DataFrame) -> pd.DataFrame:
    """Log-rank test statistic and p-value for every pair of races."""
    names = [str(r) for r in table["race"].cat.categories if (table["race"] == r).any()]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ga, gb = table[table["race"] == a], table[table["race"] == b]
            res = logrank_test(ga["time"], gb["time"], ga["fell"], gb["fell"])
            rows.append(
                {"race A": a, "race B": b, "statistic": res.test_statistic, "p": res.p_value}
            )
    return pd.DataFrame(rows)


def survival_at(fitters: Mapping[str, KaplanMeierFitter], t: float) -> pd.Series:
    """Fraction of each race still standing at time `t`."""
    if not math.isfinite(t) or t < 0:
        raise ParameterError("t must be finite and non-negative")
    return pd.Series(
        {name: float(kmf.survival_function_at_times(t).iloc[0]) for name, kmf in fitters.items()},
        name=f"S({t:g})",
    )
