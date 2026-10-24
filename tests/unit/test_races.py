import math

import numpy as np
import pandas as pd
import pytest

from corruption.model import ParameterError
from corruption.races import (
    RACES,
    SURVIVAL_COLUMNS,
    Race,
    kaplan_meier,
    median_fall_times,
    pairwise_logrank,
    survival_at,
    survival_table,
)
from corruption.stochastic import SDEConfig

FAST = SDEConfig(dt=0.1, t_max=150.0)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return survival_table(n_per_race=600, base=FAST, seed=3)


def test_table_has_one_row_per_bearer_and_the_documented_columns(table: pd.DataFrame) -> None:
    assert list(table.columns) == SURVIVAL_COLUMNS
    assert len(table) == 600 * len(RACES)
    assert list(table["race"].cat.categories) == list(RACES)


def test_censored_bearers_are_observed_at_the_horizon(table: pd.DataFrame) -> None:
    censored = table[~table["fell"]]
    assert (censored["time"] == FAST.t_max).all()
    assert (table.loc[table["fell"], "time"] <= FAST.t_max).all()


def test_only_dwarves_accumulate_greed(table: pd.DataFrame) -> None:
    by_race = table.groupby("race", observed=True)["greed"]
    assert by_race.max()["Men"] == 0.0
    assert by_race.max()["Elves"] == 0.0
    assert by_race.min()["Dwarves"] > 0.0


def test_greed_grows_with_time_held(table: pd.DataFrame) -> None:
    dwarves = table[table["race"] == "Dwarves"]
    rate = dwarves["greed"] / (dwarves["A"] * dwarves["time"])
    np.testing.assert_allclose(rate, RACES["Dwarves"].greed_share * FAST.alpha * FAST.u)


def test_men_fall_faster_than_dwarves(table: pd.DataFrame) -> None:
    fitters = kaplan_meier(table)
    medians = median_fall_times(fitters)
    assert medians["Men"] < medians["Dwarves"]
    standing = survival_at(fitters, 100.0)
    assert standing["Men"] < 0.1 < 0.5 < standing["Dwarves"]


def test_kaplan_meier_starts_at_one(table: pd.DataFrame) -> None:
    for kmf in kaplan_meier(table).values():
        assert float(kmf.survival_function_at_times(0.0).iloc[0]) == 1.0


def test_logrank_separates_men_from_dwarves(table: pd.DataFrame) -> None:
    tests = pairwise_logrank(table)
    assert len(tests) == 3
    row = tests[(tests["race A"] == "Men") & (tests["race B"] == "Dwarves")]
    assert float(row["p"].iloc[0]) < 1e-6


def test_same_seed_gives_the_same_table() -> None:
    a = survival_table(n_per_race=100, base=FAST, seed=8)
    b = survival_table(n_per_race=100, base=FAST, seed=8)
    pd.testing.assert_frame_equal(a, b)


def test_greed_diverts_part_of_the_ring_pull() -> None:
    cfg = RACES["Dwarves"].sde_config(SDEConfig())
    assert cfg.alpha == pytest.approx(SDEConfig().alpha * (1 - RACES["Dwarves"].greed_share))
    assert cfg.gamma == RACES["Dwarves"].gamma


@pytest.mark.parametrize(
    "kwargs",
    [
        {"W_mean": 0.0, "A_mean": 0.5},
        {"W_mean": 0.5, "A_mean": 1.0},
        {"W_mean": 0.5, "A_mean": 0.5, "greed_share": 1.0},
        {"W_mean": 0.5, "A_mean": 0.5, "gamma": -0.1},
        {"W_mean": 0.5, "A_mean": 0.5, "concentration": 0.0},
    ],
)
def test_invalid_race_raises(kwargs: dict[str, float]) -> None:
    with pytest.raises(ParameterError):
        Race(name="Test", **kwargs)


def test_survival_table_rejects_empty_inputs() -> None:
    with pytest.raises(ParameterError):
        survival_table({}, n_per_race=10)
    with pytest.raises(ParameterError):
        survival_table(n_per_race=0)


def test_survival_at_rejects_bad_time(table: pd.DataFrame) -> None:
    with pytest.raises(ParameterError):
        survival_at(kaplan_meier(table), math.inf)
