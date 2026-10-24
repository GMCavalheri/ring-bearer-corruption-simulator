# Project 1: Ring-Bearer Corruption Simulator

> Part of the **Dark Lord Systems** series: 10 projects that each model the same Tolkien idea, power as control over wills, with a different technique. This file is self-contained.

- **Technique:** ODE and SDE modeling, bifurcation analysis, Monte Carlo, survival analysis
- **Duration:** 3 weeks (part-time)
- **Position in the series:** suggested first project (the fastest, 2 to 3 weeks)

## Core question

How long does a bearer resist before becoming a wraith, and which small changes in character alter their whole fate?

**Lore basis:** the bearers of the Nine fell one by one, according to their native strength and initial intent ([Tolkien Forum](https://thetolkien.forum/threads/the-early-work-of-the-nine-rings.16421/)); the Dwarves did not submit ([Wikipedia](https://en.wikipedia.org/wiki/Sauron)).

## Model

Each bearer has will **W**, ambition **A** and ring usage **u(t)**. Corruption **C** evolves as an ODE with positive feedback:

```latex
\frac{dC}{dt} = \alpha \, u(t) \, A \, (1 + \beta C) - \gamma \, W \, (1 - C)
```

- Once C crosses a threshold **C\***, the bearer becomes a wraith and obeys the One.
- The feedback term **β** creates the point of no return, which is where the bifurcation appears.

## Phases

Every phase ends with something that runs and can be shown.

### Phase 1 — Deterministic model (week 1)
- Solve the ODE with `scipy.integrate.solve_ivp`.
- Plot C(t) for 9 bearers with different W and A.
- **Deliverable:** notebook + tested `corruption/model.py` module.

### Phase 2 — Bifurcation (week 2)
- Sweep A and β, map the fixed points.
- Draw the bifurcation diagram ("who resists forever and who falls").
- Reuses code from your logistic map explorer.
- **Deliverable:** bifurcation diagram + sweep module with tests.

### Phase 3 — Stochastic (week 2)
- Noisy version as a stochastic differential equation, solved with Euler–Maruyama.
- Monte Carlo run of 10,000 bearers.
- **Deliverable:** distribution of time to fall.

### Phase 4 — Races (week 3)
- Parameters per race:
  - **Dwarves:** high γ and a greed side effect (the ring amplifies wealth, not submission).
  - **Men:** high A.
- Compare survival curves with Kaplan–Meier (`lifelines`).
- **Deliverable:** survival curves per race.

### Phase 5 — App (week 3)
- Streamlit app with sliders for W, A, β and an animation of the fall.
- **Deliverable:** runnable Streamlit app.

## Extensions

- **Inverse inference:** observed fall data → estimate parameters with MCMC via PyMC.
- **Optimal control from Sauron's point of view:** which u(t) corrupts fastest without alerting the bearer.
  - This is the operations research branch **dynamic programming and Markov decision processes**: "what is the best policy over time?" In the real world the same technique drives optimal maintenance and replacement policies.

## Portfolio value

ODE and SDE modeling, bifurcation analysis, Monte Carlo, survival analysis. Survival analysis is the same technique used in component reliability.

## Connections to other projects (optional)

- **Project 5 – "Dark Lord" Game (C++):** this model becomes the game's *bearer corruption* system, where the player chooses who gets rings and when to apply pressure. Keep the model a pure function of parameters (W, A, α, β, γ, u(t)) so it can be ported.

## Stack and conventions

- **Language:** English only — code, identifiers, comments, docstrings, README, commit messages, issues, notebooks and app interface.
- **Shared stack:** Python, NumPy, SciPy, pandas, Matplotlib/Plotly, Streamlit for the interface, pytest for tests, one `pyproject.toml` for the project.
- **Project-specific:** `scipy.integrate.solve_ivp`, `lifelines` (Kaplan–Meier); PyMC for the inference extension.
- If this lives in the series monorepo, link it from the central README.

## Suggested repo layout

```
pyproject.toml
README.md
src/corruption/
    model.py          # ODE right-hand side, threshold detection
    bifurcation.py    # parameter sweeps, fixed points
    stochastic.py     # Euler–Maruyama, Monte Carlo
    races.py          # per-race parameter sets
tests/
notebooks/
app/streamlit_app.py
```

## Assumptions and copyright

- Names and lore belong to the Tolkien Estate. For non-commercial study simulations, use the theme freely without copying passages from the books.
- All parameter values (α, β, γ, W, A, C\*, per-race values) are your own invention; declare them as assumptions in the README.

## Definition of done

- [x] `corruption/model.py` with tests; C(t) plotted for 9 bearers
- [x] Bifurcation diagram over A and β
- [x] SDE + Monte Carlo of 10,000 bearers with time-to-fall distribution
- [x] Per-race Kaplan–Meier survival curves
- [x] Streamlit app with W, A, β sliders and fall animation
- [x] README with model, assumptions and screenshots
