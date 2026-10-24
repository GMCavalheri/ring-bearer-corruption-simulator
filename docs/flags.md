# Feature flags

Flags are environment variables read by `src/corruption/flags.py`. A flag is on when its variable is `1`, `true`, `yes` or `on`; unset or anything else means off. Every flag defaults to off.

| Flag | Env var | Purpose | Owner | Created | Stage | Remove by |
|---|---|---|---|---|---|---|
| `EXPERIMENTAL_SDE` | `RBCS_EXPERIMENTAL_SDE` | Shows the stochastic Monte Carlo tab in the app. It runs up to 2,000 bearers per click, so it is off until its runtime is acceptable on Streamlit Community Cloud. | @GMCavalheri | 2026-10-24 | internal (off by default) | 2027-01-31 |

Turn a flag on locally:

```bash
RBCS_EXPERIMENTAL_SDE=1 uv run streamlit run app/streamlit_app.py
```

Past the removal date, either make the behavior permanent and delete the flag, or write down here why it stays.
