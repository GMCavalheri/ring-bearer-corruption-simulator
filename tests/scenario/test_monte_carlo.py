"""Scenarios for the stochastic model: parallel determinism and noise limits."""

import numpy as np
import pytest

from corruption.model import BearerParams, analytic_fall_time
from corruption.stochastic import BLOCK_SIZE, Population, SDEConfig, monte_carlo, sample_population


def test_parallel_monte_carlo_matches_serial_run_bit_for_bit() -> None:
    pop = sample_population(3 * BLOCK_SIZE + 17, np.random.default_rng(11))
    cfg = SDEConfig(t_max=120.0)
    serial = monte_carlo(pop, cfg, seed=2026)
    for workers in (2, 4):
        parallel = monte_carlo(pop, cfg, seed=2026, n_workers=workers)
        np.testing.assert_array_equal(serial.fall_time, parallel.fall_time)


def test_a_bearer_keeps_its_fate_when_the_population_grows_by_whole_blocks() -> None:
    # Blocks own their noise stream, so the first block is unchanged when more
    # bearers are appended after it.
    big = sample_population(2 * BLOCK_SIZE, np.random.default_rng(3))
    small = big.block(0, BLOCK_SIZE)
    cfg = SDEConfig(t_max=80.0)
    np.testing.assert_array_equal(
        monte_carlo(small, cfg, seed=9).fall_time,
        monte_carlo(big, cfg, seed=9).fall_time[:BLOCK_SIZE],
    )


@pytest.mark.parametrize("sigma", [1e-3, 1e-4])
def test_vanishing_noise_recovers_the_deterministic_fall_time(sigma: float) -> None:
    n = 200
    pop = Population(W=np.full(n, 0.5), A=np.full(n, 0.8), c0=np.full(n, 0.05))
    result = monte_carlo(pop, SDEConfig(sigma=sigma, dt=0.01, t_max=100.0), seed=1)
    exact = analytic_fall_time(BearerParams(W=0.5, A=0.8))
    assert result.fell.all()
    assert float(np.mean(result.fall_time)) == pytest.approx(exact, rel=2e-3)


def test_noise_can_topple_a_bearer_who_would_resist_forever() -> None:
    n = 2000
    pop = Population(W=np.full(n, 0.9), A=np.full(n, 0.3), c0=np.full(n, 0.0))
    quiet = monte_carlo(pop, SDEConfig(sigma=0.0, t_max=300.0), seed=5)
    noisy = monte_carlo(pop, SDEConfig(sigma=0.05, t_max=300.0), seed=5)
    assert quiet.fraction_fallen == 0.0
    assert 0.0 < noisy.fraction_fallen < 1.0
