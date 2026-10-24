"""Stochastic corruption: Euler–Maruyama and Monte Carlo over many bearers.

The noisy model adds a Wiener term to the deterministic drift:

    dC = [α·u·A·(1 + β·C) − γ·W·(1 − C)] dt + σ dB_t

C is reflected at 0 (corruption cannot go negative) and absorbed at C* (the
fall). Bearers are simulated as vectors, one entry per bearer.

Reproducibility: bearers are split into fixed blocks of `BLOCK_SIZE`, and each
block draws its noise from its own child of `SeedSequence(seed)`. The result
therefore depends only on the seed, never on how many workers run the blocks.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np

from corruption.model import FloatArray, ParameterError

BLOCK_SIZE = 1024


@dataclass(frozen=True)
class Population:
    """Per-bearer will W, ambition A and initial corruption C0 (equal-length arrays)."""

    W: FloatArray
    A: FloatArray
    c0: FloatArray

    def __post_init__(self) -> None:
        shapes = {np.shape(self.W), np.shape(self.A), np.shape(self.c0)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ParameterError("W, A and c0 must be 1-D arrays of the same length")
        if len(self.W) == 0:
            raise ParameterError("population must not be empty")
        for name in ("W", "A", "c0"):
            arr = getattr(self, name)
            if not np.all(np.isfinite(arr)) or np.any(arr < 0):
                raise ParameterError(f"{name} must be finite and non-negative")

    def __len__(self) -> int:
        return len(self.W)

    def block(self, start: int, stop: int) -> Population:
        return Population(W=self.W[start:stop], A=self.A[start:stop], c0=self.c0[start:stop])


@dataclass(frozen=True)
class SDEConfig:
    """Parameters shared by every bearer in a Monte Carlo run."""

    alpha: float = 0.02
    beta: float = 3.0
    gamma: float = 0.02
    c_star: float = 0.9
    sigma: float = 0.02
    u: float = 1.0
    dt: float = 0.05
    t_max: float = 200.0

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "c_star", "sigma", "u", "dt", "t_max"):
            if not math.isfinite(getattr(self, name)):
                raise ParameterError(f"{name} must be finite")
        if self.alpha <= 0:
            raise ParameterError("alpha must be positive")
        if min(self.beta, self.gamma, self.sigma, self.u) < 0:
            raise ParameterError("beta, gamma, sigma and u must be non-negative")
        if not 0 < self.c_star <= 1:
            raise ParameterError("c_star must be in (0, 1]")
        if self.dt <= 0 or self.t_max <= 0 or self.dt > self.t_max:
            raise ParameterError("need 0 < dt <= t_max")

    @property
    def n_steps(self) -> int:
        return math.ceil(self.t_max / self.dt - 1e-9)


@dataclass(frozen=True)
class MonteCarloResult:
    """Fall time per bearer (`inf` if still standing at the horizon, i.e. censored)."""

    fall_time: FloatArray
    horizon: float

    @property
    def fell(self) -> np.ndarray:
        return np.isfinite(self.fall_time)

    @property
    def fraction_fallen(self) -> float:
        return float(self.fell.mean())

    def observed_time(self) -> FloatArray:
        """Fall time, or the horizon for censored bearers (survival-analysis input)."""
        return np.where(self.fell, self.fall_time, self.horizon)


def beta_from_mean(
    rng: np.random.Generator, mean: float, concentration: float, size: int
) -> FloatArray:
    """Beta-distributed values in (0, 1) with the given mean and concentration κ = a + b."""
    if not 0 < mean < 1 or concentration <= 0:
        raise ParameterError("need 0 < mean < 1 and concentration > 0")
    return rng.beta(mean * concentration, (1 - mean) * concentration, size)


def sample_population(
    n: int,
    rng: np.random.Generator,
    *,
    W_mean: float = 0.6,
    A_mean: float = 0.75,
    c0_mean: float = 0.05,
    concentration: float = 20.0,
) -> Population:
    """Draw `n` bearers with Beta-distributed W, A and C0 around the given means."""
    if n <= 0:
        raise ParameterError("n must be positive")
    return Population(
        W=beta_from_mean(rng, W_mean, concentration, n),
        A=beta_from_mean(rng, A_mean, concentration, n),
        c0=beta_from_mean(rng, c0_mean, concentration, n),
    )


def _steps(
    pop: Population, cfg: SDEConfig, rng: np.random.Generator
) -> Iterator[tuple[float, FloatArray, FloatArray]]:
    """Yield (t, C, fall_time) after each Euler–Maruyama step, starting at t = 0.

    Only bearers still standing are stepped, so a run gets cheaper as they fall.
    The yielded arrays are updated in place; copy them to keep a snapshot.
    """
    if np.any(pop.c0 >= cfg.c_star):
        raise ParameterError("every c0 must be below c_star")
    c = pop.c0.astype(np.float64, copy=True)
    fall_time = np.full(len(pop), np.inf)
    drive = cfg.alpha * cfg.u * pop.A
    restore = cfg.gamma * pop.W
    sqrt_dt = math.sqrt(cfg.dt)
    idx = np.arange(len(pop))
    yield 0.0, c, fall_time
    for k in range(cfg.n_steps):
        if idx.size == 0:
            return
        t = k * cfg.dt
        c_old = c[idx]
        drift = drive[idx] * (1.0 + cfg.beta * c_old) - restore[idx] * (1.0 - c_old)
        noise = rng.standard_normal(idx.size)
        c_new = np.abs(c_old + drift * cfg.dt + cfg.sigma * sqrt_dt * noise)
        crossed = c_new >= cfg.c_star
        # Place the fall inside the step by linear interpolation.
        frac = (cfg.c_star - c_old[crossed]) / (c_new[crossed] - c_old[crossed])
        fall_time[idx[crossed]] = t + cfg.dt * np.clip(frac, 0.0, 1.0)
        c[idx] = np.where(crossed, cfg.c_star, c_new)
        idx = idx[~crossed]
        yield t + cfg.dt, c, fall_time


def simulate_paths(
    pop: Population, cfg: SDEConfig, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Full sample paths for a small population.

    Returns (t, C, fall_time) where C has shape (len(t), len(pop)) and is held
    at C* after a bearer falls.
    """
    times: list[float] = []
    paths: list[FloatArray] = []
    fall_time = np.full(len(pop), np.inf)
    for t, c, current_fall_time in _steps(pop, cfg, rng):
        times.append(t)
        paths.append(c.copy())
        fall_time = current_fall_time
    return np.asarray(times), np.vstack(paths), fall_time.copy()


def _run_block(cfg: SDEConfig, pop: Population, seed: np.random.SeedSequence) -> FloatArray:
    fall_time = np.full(len(pop), np.inf)
    for _, _, current_fall_time in _steps(pop, cfg, np.random.default_rng(seed)):
        fall_time = current_fall_time
    return fall_time.copy()


def monte_carlo(
    pop: Population,
    cfg: SDEConfig,
    seed: int,
    *,
    n_workers: int = 1,
) -> MonteCarloResult:
    """Simulate every bearer until the fall or `cfg.t_max`.

    Blocks of `BLOCK_SIZE` bearers run on `n_workers` processes. The result is
    identical for any `n_workers` given the same seed.
    """
    if n_workers < 1:
        raise ParameterError("n_workers must be at least 1")
    starts = range(0, len(pop), BLOCK_SIZE)
    seeds = np.random.SeedSequence(seed).spawn(len(starts))
    blocks = [pop.block(s, s + BLOCK_SIZE) for s in starts]
    run = partial(_run_block, cfg)
    if n_workers == 1:
        parts = list(map(run, blocks, seeds))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            parts = list(pool.map(run, blocks, seeds))
    return MonteCarloResult(fall_time=np.concatenate(parts), horizon=cfg.t_max)
