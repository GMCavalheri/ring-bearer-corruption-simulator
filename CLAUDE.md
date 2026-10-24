# Ring-Bearer Corruption Simulator

Models how a ring-bearer's corruption C grows under ring use, finds the point of no return, and compares how fast different races fall. Part 1 of the Dark Lord Systems series (see `PLAN.md`).

## Architecture overview

- **Components**: pure numerical core in `src/corruption/` → notebooks in `notebooks/` and the Streamlit app in `app/` consume it. There is no database, network service or shared mutable state.
- **Core domain operation**: integrate dC/dt = αuA(1+βC) − γW(1−C) and detect the moment C crosses C* (the fall). Lives in `src/corruption/model.py`. Invariant: C stays in [0, C*], and the numeric fall time agrees with the closed form for constant u.
- **Modules**: `model.py` (ODE, closed form, fixed point), `bifurcation.py` (A × β sweeps), `stochastic.py` (Euler–Maruyama, Monte Carlo), `races.py` (per-race parameters, Kaplan–Meier), `plotting.py` (Matplotlib figures for notebooks/README), `interactive.py` (Plotly figures for the app), `flags.py` (env-var feature flags).
- **Figures**: notebooks write PNGs to `docs/figures/`; the README embeds them. Re-run a notebook after changing defaults or the README numbers go stale.
- **Delivery pipeline**: commit → push to `main` → GitHub Actions CI (`lint`, `unit` on Python 3.11–3.13, `scenario`, `notebooks`, `app-smoke`, `security`). There is no deploy target; releases are git tags (`v1.0.0`).
- **Feature flags**: env vars `RBCS_EXPERIMENTAL_*`, read in `src/corruption/flags.py`, registry in `docs/flags.md`.

## Conventions

- English only: code, comments, docstrings, notebooks, commit messages, app text.
- `model.py` stays a pure function of (W, A, α, β, γ, u(t)) with no plotting or I/O, so Project 5 can port it to C++.
- Math names follow the paper notation (W, A, C, α, β, γ); ruff's N803/N806 are disabled for that reason.
- Randomness always goes through an explicit `numpy.random.Generator` or seed argument. No global `np.random.*` calls.
- Tests: `tests/unit` (fast, pure) and `tests/scenario` (numerical scenarios: solver agreement, reproducibility, failure on bad input). Scenario tests are named after the scenario.
- Every parameter value is an invented assumption and must appear in the README assumptions table.
- Notebooks are committed executed (outputs included) so GitHub renders them; CI re-executes them with nbmake.
- `docs/lessons/` is a six-lesson course, one lesson per phase plus V&V. Every number in it was computed by the package. When a default, a preset or a seeded result changes, rerun `make_figures.py` and the notebooks and update the lesson text. Relative links are checked by `tests/unit/test_docs_links.py`.

## Commands

Only commands verified to work. Update this table when a command changes.

| Task | Command |
|---|---|
| Install | `uv sync` |
| Lint / format / typecheck | `uv run ruff check . && uv run ruff format --check . && uv run mypy src app` |
| Unit tests + coverage gate | `uv run pytest tests/unit --cov=corruption` |
| Scenario tests | `uv run pytest tests/scenario` |
| All tests | `uv run pytest` |
| Notebooks (execute all) | `uv run pytest --nbmake notebooks/` |
| App smoke tests (headless) | `uv run pytest tests/app` |
| Run the app | `uv run streamlit run app/streamlit_app.py` |
| Run the app with experimental tab | `RBCS_EXPERIMENTAL_SDE=1 uv run streamlit run app/streamlit_app.py` |
| Lesson figures | `uv run python docs/lessons/make_figures.py` |
| Dependency audit | `uv export --no-emit-project --all-groups --format requirements-txt -o /tmp/req.txt -q && uv run pip-audit -r /tmp/req.txt --disable-pip --strict` |

## Non-obvious gotchas

- `uv sync` fails if `README.md` is missing, because hatchling validates the `readme` field.
- `astral-sh/setup-uv` publishes no moving major tag (`@v10` fails to resolve). Pin the full version, e.g. `@v10.3.2`; dependabot bumps it.
- `gitleaks-action` on pull requests needs `pull-requests: read` to list PR commits; without it the security job fails with 403.
- Monte Carlo parallelism uses processes, not threads: the Euler–Maruyama loop is many small NumPy calls, and threads ran 2.5x slower than serial because of the GIL.
- Monte Carlo results depend on `BLOCK_SIZE` in `stochastic.py` (each block owns a seed stream). Changing it changes every seeded result.
- Never form C_u = −b/a to compute fall times: it overflows for tiny W or a subnormal C0. Use `fall_time_from_coefficients` (log-difference form) and decide fate from the sign of a·C0 + b. Hypothesis found both cases.
- When piping test output (`pytest | tail`), run `set -o pipefail` first, or a failure exits 0.
- Streamlit only hot-reloads the app script, not `src/corruption` modules: restart the server after editing the package.
- Plotly animations crawl when the browser tab is hidden or in the background (requestAnimationFrame throttling); that is not an app bug.
- `tests/app` uses `streamlit.testing.v1.AppTest`, which runs the real script headless. Use it instead of mocking Streamlit.
- lifelines is locked at 0.30.0 on purpose: 0.30.1+ require `pandas<3.0`, so uv picks 0.30.0 to allow pandas 3. Kaplan–Meier and log-rank results were checked identical under pandas 2.3.3 + lifelines 0.30.3. Its Pandas4Warning noise comes from inside lifelines.
- `pip-audit` on the live environment flags the local package as "not on PyPI". Audit the exported lock (`uv export --no-emit-project`) instead.

## Never do in this repo

- Never push code that fails lint, typecheck or tests locally.
- Never use unseeded randomness in library code or tests.
- Never add plotting, file or UI code to `model.py`.
- Never copy passages from Tolkien's books; use names and themes only.
- Never commit secrets. Keep `.env` out of git.
