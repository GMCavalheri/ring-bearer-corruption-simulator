## What changes

<!-- One or two sentences. -->

## Claims and evidence

Every claim needs evidence: a test name, a CI run link, a figure or a log excerpt.

| Claim | Evidence |
|---|---|
| | |

## Checklist

- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy src` pass
- [ ] `uv run pytest` passes, and new behavior has a test that fails without the change
- [ ] Notebooks still run (`uv run pytest --nbmake notebooks/`)
- [ ] New parameter values are listed as assumptions in the README
- [ ] Experimental app behavior sits behind a flag in `docs/flags.md`
