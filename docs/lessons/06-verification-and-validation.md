# Lesson 6 · How this project knows it is right

> *Every parameter in this project is invented, so no measurement can prove the model right. That
> makes checking the code more important, not less. This lesson collects the checks, and the
> mistakes they caught. Every one of these mistakes actually happened while building the
> project.*

**Code:** [`tests/`](../../tests) · [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) ·
**Previous:** [Lesson 5 — From model to Streamlit app](05-interactive-app.md) ·
**Next:** [Course index](README.md)

## Learning objectives

By the end of this lesson you should be able to

1. distinguish verification ("are we solving the equations right?") from validation ("are we
   solving the right equations?");
2. build a hierarchy of checks for a numerical model, from exact identities to end-to-end runs;
3. use property-based testing to find inputs you would never have thought of;
4. say honestly what can and cannot be validated when parameters are assumptions;
5. make a numerical project reproducible down to the last digit;
6. plan the checks for a new feature before writing it.

---

## 1. Two different questions

- **Verification:** does the code solve the model correctly? This is mathematics and software
  engineering, and it *can* be settled: against closed forms, convergence rates and invariants.
- **Validation:** does the model describe reality? This needs data. Here the "reality" is
  Tolkien's legendarium, and every rate, threshold and distribution is an invented assumption.

So this project can be fully verified and only weakly validated, and it says so. The README lists
every assumed value. The lessons call the results "under the assumptions", never predictions.

## 2. A hierarchy of verification checks

Each level catches mistakes the levels below cannot.

| Level | What it checks | Example in this project | Where |
|---|---|---|---|
| 1. Exact identities | algebra the code must satisfy exactly | `rhs` equals $aC + b$; $C_u$ has zero drift; $C_u$ = 0 at $A_c$ for every β | `tests/unit/test_model.py`, `test_bifurcation.py` |
| 2. Closed form vs numeric | an independent route to the same number | `solve_ivp` fall time = closed form to 1e-6, on 40 random bearers | `tests/scenario/test_solver_agreement.py` |
| 3. Convergence order | the error shrinks at the rate theory predicts | Euler–Maruyama at σ = 0: error halves when dt halves | `tests/unit/test_stochastic.py` |
| 4. Properties over random inputs | invariants for *any* valid input | C stays in [0, C\*]; more ambition never delays the fall; more will never hastens it | Hypothesis tests in `test_model.py` |
| 5. Cross-implementation agreement | two codes for one quantity | vectorized `sweep` = scalar closed form = 100,000-year ODE runs, on every grid cell | `test_bifurcation_agrees_with_simulation.py` |
| 6. Statistical limits | the stochastic model reduces to the deterministic one | mean fall time → ODE fall time as σ → 0 | `tests/scenario/test_monte_carlo.py` |
| 7. Reproducibility | the same seed gives the same bits | serial = 2 workers = 4 workers; a block is stable when blocks are added | `test_monte_carlo.py` |
| 8. End to end | the deliverables actually run | notebooks executed by nbmake; the app run headless by AppTest; lesson figures rebuilt; doc links resolve | CI jobs `notebooks`, `app-smoke` |
| 9. Static and supply chain | the code and its dependencies | ruff, `mypy --strict`, gitleaks, pip-audit on the lock file | CI jobs `lint`, `security` |

A coverage gate (90%, currently 99%) sits on top. Coverage only says which lines ran, not
whether they are right; the levels above say that.

### Why closed forms are worth so much

Level 2 is the strongest check in the project, and it exists only because Lesson 1 noticed the
equation is linear. A closed form turns "the plot looks plausible" into "the error is 9 × 10⁻⁸
years". The same closed form then served as the oracle for the solver tolerance (Lesson 1), the
Euler–Maruyama order (Lesson 3) and the vectorized sweep (Lesson 2). **When a model has an exact
special case, build the test suite around it.**

## 3. Property-based testing found what examples could not

An example-based test checks inputs you chose. A **property-based test** states a rule and lets a
library (Hypothesis) search for inputs that break it, including denormal floats, zeros and
boundaries. Two of its finds were real bugs:

1. **Subnormal will.** With β = 0 and W = 2.2 × 10⁻³¹³, $a = \gamma W$ = 4.5 × 10⁻³¹⁵ is a
   subnormal number and $C_u = -b/a$ overflows to −∞. The textbook fall time then evaluates
   ∞/∞ = **NaN**. The property "more ambition never delays the fall" failed with `nan <= inf`.
2. **Subnormal starting corruption.** Hypothesis kept searching after the first fix. With
   C0 = 2.2 × 10⁻³⁰⁹ just above $C_u$ = 0, the ratio $(C^* - C_u)/(C_0 - C_u)$ overflows. The
   textbook formula then returns ∞ ("never falls") for a bearer who actually falls after 71,059
   years. The first rewrite turned the same overflow into NaN, and "more will never hastens the
   fall" failed with `nan >= 90`.

Both landed in one commit (`38fea31`): the fall time became
$[\ln(aC^* + b) - \ln(aC_0 + b)]/a$ with a linear fallback, and the fate became the sign of
$aC_0 + b$. Neither form ever builds $C_u$. Each bug got a regression test with the exact failing
input, so it can never come back silently.

No one would write `W=5e-324` in an example test. That is the point of property-based testing.

## 4. Case studies: what else went wrong, and what caught it

| # | Mistake | Caught by | Fix |
|---|---|---|---|
| 1 | PLAN.md says β "creates the point of no return" | deriving $A_c$ (Lesson 2) | kept the model, corrected the README |
| 2 | NaN and ∞ fall times at extreme inputs (Section 3) | property tests | log-difference fall time |
| 3 | a commit with a failing test was pushed | the next local run | `set -o pipefail` before piping test output |
| 4 | `DeprecationWarning: ... use of fork() may lead to deadlocks` | reading test warnings | process pool with the `spawn` start method |
| 5 | `astral-sh/setup-uv@v10` did not exist | first CI run, red in 7 seconds | pin the full version, `@v10.3.2` |
| 6 | gitleaks failed with 403 on the first Dependabot PR | CI on that PR | `pull-requests: read` for the security job |
| 7 | pandas 3 upgrade silently downgraded lifelines | reading the lock-file diff | ran the survival analysis on both stacks: identical to every digit |
| 8 | RK45 stepped over a short usage pulse | writing Lesson 1 | documented `max_step`; the pulse is a pitfall with numbers |
| 9 | draft exercise answers that were wrong | computing every answer with the code | rewrote the exercises |
| 10 | the regression test for bug 2 did not reproduce it | rerunning the *old* code against the test while writing this lesson | moved the test to the real failing input |

A few deserve a closer look.

**3. The red commit.** The check was `uv run pytest -q 2>&1 | tail -1`. A pipeline's exit status is
that of its *last* command, `tail`, which succeeds. The failing test was printed and ignored, and the
commit was pushed. The fix is one line, `set -o pipefail`, now recorded in `CLAUDE.md`. **A check
that cannot fail is not a check.**

**4. Fork after threads.** The first parallel Monte Carlo used the default `fork` start method.
NumPy's BLAS and Streamlit start threads, and forking a multi-threaded process can deadlock the
child. Python only *warned*. The warning appeared in the test summary, and switching to `spawn`
removed it. The cost: scripts need a `__main__` guard (Lesson 3).

**7. The dependency that moved backwards.** Dependabot's PR said "bump pandas and lifelines". The
lock file showed lifelines going *down*, from 0.30.3 to 0.30.0, because newer lifelines require
pandas < 3. CI was green, but green only says the tests pass. Before merging, the full race
survival analysis ran under both stacks. Medians, S(t) and log-rank statistics matched to every
digit.

**9. Wrong answers in a draft.** A draft exercise for Lesson 3 asked students to reflect a step with
Z = −1.2; the code showed that step stays positive, so nothing is reflected. A draft of notebook 01
claimed W = 0.81 saves bearer 9; the code showed it still falls. **Every number in these lessons was
computed by the package, then written down, never the other way round.**

**10. A regression test that could not fail.** The first regression test for the subnormal-will
bug used W = 5 × 10⁻³²⁴, the smallest positive double. But 0.02 × 5 × 10⁻³²⁴ underflows to exactly
0, so the original code took its separate `a == 0` branch, which never had the bug. The test passed
on the buggy version too. Running the commit-`06b14c1` implementation against the test exposed it:
85.0 years, a pass. The test now uses W = 2.2250738585 × 10⁻³¹³, where the original returns NaN, and
it asserts $\gamma W > 0$ so the input cannot silently degrade again. **A regression test must be
shown to fail on the code it guards against.**

## 5. Validation without data

With invented parameters you cannot validate quantitatively, but you can still do three honest
things:

- **Face validity.** The behavior must match the source qualitatively: the Nine fall one by one, at
  different times; Dwarves do not submit; strong will with low ambition resists. The model was tuned
  to reproduce these facts, so they show it *can* tell the story, not that it is true.
- **Sensitivity over precision.** Results are reported as patterns ("Dwarves survive because of the
  greed diversion: without it their median drops to 53 years") rather than as precise forecasts.
- **Labelling.** The README separates three kinds of number: assumptions (α, β, γ, σ, race means),
  results computed from them, and mathematical facts true for *any* parameters ($A_c$ does not depend
  on β). Only the last kind holds outside this project.

## 6. Reproducibility

Anyone with the repository should get the same numbers, digit for digit:

- **Pinned environment.** `uv.lock` fixes every package version. CI runs `uv sync` with
  `UV_FROZEN=1`.
- **Explicit randomness.** No global `np.random` calls. Every random function takes a `Generator`
  or a seed, and Monte Carlo blocks own `SeedSequence` children (Lesson 3).
- **Executed artifacts are rebuilt in CI.** The notebooks are committed with outputs so GitHub
  shows them. CI re-executes them, and also rebuilds every lesson figure, so neither can rot
  silently.
- **Numbers in prose come from code.** README tables, notebook narratives and these lessons were
  written from printed results. When defaults change, the notebooks and `make_figures.py` must be
  rerun and the text updated.

## 7. Where this model ends

Verification says the code solves the model. These are the limits of the model itself:

- **One state variable.** Real corruption would interact with fear, loyalty or the ring's own will.
- **Linear in C.** No saddle-node is possible, so there is no stable state of "partial corruption"
  (Lesson 2, Exercise 3 shows what that would need).
- **Constant traits.** W, A and the rates never change during a life.
- **Additive Gaussian noise.** No sudden shocks (a single catastrophic temptation), and the noise
  does not depend on how corrupted the bearer already is.
- **Races as distributions.** Races differ only in their parameter distributions and γ, plus the
  Dwarves' greed term.

Each limit is a modeling choice, not a bug. Each would also be a natural extension.

## 8. Worked example: planning checks for a new feature

Suppose you add the saddle-node variant from Lesson 2, Exercise 3: resistance
$4\gamma W\,C(1 - C)$. Plan the checks before writing the code:

1. **Identities.** The new right-hand side equals $rC^2 + (q - r)C + p$. Both interior equilibria
   have zero drift. At $A_{sn}$ = 4/15 they coincide at C = 1/3.
2. **Stability.** The derivative of the drift is negative at the lower equilibrium and positive at
   the upper one.
3. **Closed form vs numeric.** With constant coefficients the quadratic ODE is separable, so it has
   a closed form too (partial fractions). Compare `solve_ivp` against it as in Level 2.
4. **Properties.** C stays in [0, C\*], and more ambition never delays the fall.
5. **Cross-check with the old model.** The two models must agree wherever their resistance terms
   agree. If no such region exists, test the *qualitative* difference instead: the old model has no
   interior stable state, and the new one does below $A_{sn}$.
6. **End to end.** A notebook section with the saddle-node diagram runs in CI. If the app shows it,
   put the tab behind a new flag and add `AppTest` cases for both flag states.

## 9. Common pitfalls

- **Green CI as proof.** CI proves the tests passed, nothing more. Read diffs, warnings and lock
  files.
- **Tests that cannot fail.** Piped exit codes, `assert` on a value that is always true, a scenario
  that never triggers. Break the code on purpose once and watch the test go red.
- **Validating against your own tuning.** If parameters were chosen to reproduce a fact,
  reproducing it is not evidence.
- **Ignoring warnings.** The fork deadlock risk was only a warning.
- **Numbers typed from memory.** Compute every number you publish.

## 10. Exercises

1. Classify each check as verification or validation: (a) the Euler–Maruyama error halves when dt
   halves; (b) the Nine fall at different times; (c) Kaplan–Meier S(0) = 1; (d) Men fall faster than
   Dwarves.
2. Write a Hypothesis property for `sweep()` that would have caught the subnormal-W bug in the
   vectorized code path too.
3. The test `test_more_will_never_hastens_the_fall` compares `analytic_fall_time` values. Why is it a
   stronger test than comparing against three hand-picked bearers?
4. Name two checks in this project that would still pass if `BLOCK_SIZE` were changed, and one that
   would fail.
5. A colleague proposes removing the closed-form tests "because `solve_ivp` is a well-tested
   library". What would the project lose?

<details>
<summary>Answers</summary>

1. (a) **Verification**: it checks the numerical method against its theory. (b) **Validation**
   (face validity) against the lore, weakened because the parameters were tuned to get it. (c)
   **Verification** of an identity of the estimator. (d) Mostly **validation** (it matches the
   lore's story about Men), but in this project it is also a regression check on the race presets.
2. For example: given W in [0, 1] (Hypothesis generates subnormals near 0), A and β in their ranges and
   C0 in [0, 0.5], `result = sweep(base, [A], [beta])`; assert that `result.fall_time` is never NaN,
   and that it equals `analytic_fall_time(BearerParams(W=W, A=A, beta=beta, c0=C0))` where finite.
   Both functions now share `fall_time_from_coefficients`, so the property also guards the shared
   helper.
3. It quantifies over *all* valid bearers, and Hypothesis searches for counterexamples, including
   denormal floats and exact boundaries. Three hand-picked bearers would all have had ordinary
   magnitudes, and the NaN bugs would have shipped.
4. Pass: serial equals parallel (both use the same blocks), and the σ → 0 limit (a statistical
   tolerance). Fail: any test that pins a specific seeded value, e.g. a notebook number such as
   "99.62% fallen", or a regression test with an exact seeded array. The *statistics* would be
   unchanged; the individual bits would not.
5. `solve_ivp` being correct does not make *our use of it* correct: the clamp, the event direction,
   the tolerances, `max_step` with pulsed usage, and the definition of the fall are all ours. The
   closed form also verifies the vectorized sweep and the Euler–Maruyama order, and it pinned down the
   NaN fixes. Without it, the project would only know that its numbers look plausible.

</details>

## Key takeaways

- Verification asks whether the code solves the model; validation asks whether the model is right.
  With invented parameters, verify fully and validate honestly.
- Build checks in levels: identities, closed forms, convergence rates, properties, cross-checks,
  statistical limits, reproducibility, end-to-end runs.
- Exact special cases are the best oracles. Look for them before testing anything else.
- Property-based testing finds the inputs you would never write. Pin every find as a regression test.
- Green CI is necessary, not sufficient: read warnings, diffs and lock files.
- Compute every number you publish.

## Further reading

- W. L. Oberkampf & C. J. Roy, *Verification and Validation in Scientific Computing*, Ch. 1–2
  (terminology) and Ch. 5 (code verification by order of accuracy).
- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Ch. 2.
- D. R. MacIver et al., "Hypothesis: A new approach to property-based testing", *Journal of Open
  Source Software* 4(43), 2019.
- G. Wilson et al., "Good enough practices in scientific computing", *PLOS Computational Biology*
  13(6), 2017.
