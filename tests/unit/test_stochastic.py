import math

import numpy as np
import pytest

from corruption.model import BearerParams, ParameterError, analytic_fall_time
from corruption.stochastic import (
    MonteCarloResult,
    Population,
    SDEConfig,
    beta_from_mean,
    monte_carlo,
    sample_population,
    simulate_paths,
)


def single(W: float, A: float, c0: float = 0.05) -> Population:
    return Population(W=np.array([W]), A=np.array([A]), c0=np.array([c0]))


def test_without_noise_em_tracks_the_closed_form_fall_time() -> None:
    p = BearerParams(W=0.5, A=0.8)
    result = monte_carlo(single(0.5, 0.8), SDEConfig(sigma=0.0, dt=0.01), seed=0)
    assert result.fall_time[0] == pytest.approx(analytic_fall_time(p), rel=1e-3)


def test_without_noise_em_error_is_first_order_in_dt() -> None:
    exact = analytic_fall_time(BearerParams(W=0.5, A=0.8))
    errors = [
        abs(monte_carlo(single(0.5, 0.8), SDEConfig(sigma=0.0, dt=dt), seed=0).fall_time[0] - exact)
        for dt in (0.2, 0.1, 0.05)
    ]
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    assert all(1.7 < r < 2.3 for r in ratios)


def test_without_noise_a_resister_never_falls() -> None:
    result = monte_carlo(single(0.9, 0.3), SDEConfig(sigma=0.0), seed=0)
    assert not result.fell.any()
    assert result.fraction_fallen == 0.0


def test_same_seed_same_result_different_seed_different_result() -> None:
    pop = sample_population(500, np.random.default_rng(0))
    cfg = SDEConfig(t_max=100.0)
    a = monte_carlo(pop, cfg, seed=1)
    b = monte_carlo(pop, cfg, seed=1)
    c = monte_carlo(pop, cfg, seed=2)
    np.testing.assert_array_equal(a.fall_time, b.fall_time)
    assert not np.array_equal(a.fall_time, c.fall_time)


def test_paths_are_reflected_at_zero_and_held_at_the_threshold() -> None:
    pop = Population(W=np.array([0.9, 0.3]), A=np.array([0.2, 0.95]), c0=np.array([0.0, 0.1]))
    t, paths, fall_time = simulate_paths(
        pop, SDEConfig(sigma=0.05, t_max=80.0), np.random.default_rng(3)
    )
    assert paths.shape == (len(t), 2)
    assert paths.min() >= 0.0
    assert paths.max() <= 0.9
    assert math.isfinite(fall_time[1])
    after_fall = t > fall_time[1]
    assert np.all(paths[after_fall, 1] == 0.9)


def test_fall_times_lie_inside_the_horizon() -> None:
    pop = sample_population(300, np.random.default_rng(1))
    result = monte_carlo(pop, SDEConfig(t_max=30.0), seed=4)
    fell = result.fall_time[result.fell]
    assert np.all((fell > 0) & (fell <= 30.0))


def test_observed_time_uses_the_horizon_for_censored_bearers() -> None:
    result = MonteCarloResult(fall_time=np.array([5.0, np.inf]), horizon=50.0)
    np.testing.assert_array_equal(result.observed_time(), [5.0, 50.0])
    np.testing.assert_array_equal(result.fell, [True, False])


def test_sample_population_means_are_close_to_targets() -> None:
    pop = sample_population(20_000, np.random.default_rng(5), W_mean=0.4, A_mean=0.7)
    assert pop.W.mean() == pytest.approx(0.4, abs=0.01)
    assert pop.A.mean() == pytest.approx(0.7, abs=0.01)
    assert len(pop) == 20_000


@pytest.mark.parametrize(("mean", "concentration"), [(0.0, 10.0), (1.0, 10.0), (0.5, 0.0)])
def test_beta_from_mean_rejects_bad_shape(mean: float, concentration: float) -> None:
    with pytest.raises(ParameterError):
        beta_from_mean(np.random.default_rng(0), mean, concentration, 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": -0.1},
        {"dt": 0.0},
        {"dt": 10.0, "t_max": 5.0},
        {"alpha": 0.0},
        {"c_star": 1.5},
        {"u": -1.0},
        {"t_max": math.inf},
    ],
)
def test_sde_config_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ParameterError):
        SDEConfig(**kwargs)


def test_population_rejects_mismatched_or_invalid_arrays() -> None:
    with pytest.raises(ParameterError):
        Population(W=np.ones(3), A=np.ones(2), c0=np.zeros(3))
    with pytest.raises(ParameterError):
        Population(W=np.array([]), A=np.array([]), c0=np.array([]))
    with pytest.raises(ParameterError):
        Population(W=np.array([-0.1]), A=np.array([0.5]), c0=np.array([0.0]))
    with pytest.raises(ParameterError):
        Population(W=np.ones((2, 2)), A=np.ones((2, 2)), c0=np.zeros((2, 2)))


def test_bearer_starting_past_the_threshold_is_rejected() -> None:
    with pytest.raises(ParameterError):
        monte_carlo(single(0.5, 0.5, c0=0.95), SDEConfig(), seed=0)


def test_invalid_worker_count_and_population_size() -> None:
    with pytest.raises(ParameterError):
        monte_carlo(single(0.5, 0.5), SDEConfig(), seed=0, n_workers=0)
    with pytest.raises(ParameterError):
        sample_population(0, np.random.default_rng(0))
