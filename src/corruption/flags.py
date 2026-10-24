"""Environment-variable feature flags.

A flag is on when `RBCS_<NAME>` is set to 1, true, yes or on (any case).
Anything else, including an unset variable, means the flag's default, which is
always off. Unset the variable to turn a flag back off: no redeploy needed.
Registry with owners and removal dates: docs/flags.md.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

PREFIX = "RBCS_"
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Flag:
    name: str
    description: str
    default: bool = False

    @property
    def env_var(self) -> str:
        return PREFIX + self.name


FLAGS: dict[str, Flag] = {
    "EXPERIMENTAL_SDE": Flag(
        name="EXPERIMENTAL_SDE",
        description="Show the stochastic Monte Carlo tab in the Streamlit app.",
    ),
}


def is_enabled(name: str, env: Mapping[str, str] | None = None) -> bool:
    """Whether flag `name` is on. Raises KeyError for an unknown flag."""
    flag = FLAGS[name]
    environ = os.environ if env is None else env
    raw = environ.get(flag.env_var)
    if raw is None:
        return flag.default
    return raw.strip().lower() in _TRUTHY
