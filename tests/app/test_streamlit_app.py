"""Smoke tests: run the real Streamlit script headless and drive its widgets."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def slider(at: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(s for s in at.slider if s.label == label)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.delenv("RBCS_EXPERIMENTAL_SDE", raising=False)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_default_bearer_falls(app: AppTest) -> None:
    assert metric(app, "Fate") == "Falls"
    assert "years" in metric(app, "Becomes a wraith after")


def test_raising_will_and_lowering_ambition_flips_the_fate(app: AppTest) -> None:
    slider(app, "Will W").set_value(0.95)
    slider(app, "Ambition A").set_value(0.3)
    app.run()
    assert not app.exception
    assert metric(app, "Fate") == "Resists"
    assert metric(app, "Becomes a wraith after") == "never"


def test_slow_fall_beyond_the_horizon_is_reported(app: AppTest) -> None:
    slider(app, "Horizon (years)").set_value(20)
    app.run()
    assert metric(app, "Fate") == "Falls (later)"


def test_stochastic_tab_is_hidden_when_the_flag_is_off(app: AppTest) -> None:
    assert [t.label for t in app.tabs] == ["Fall animation", "Bifurcation"]


def test_stochastic_tab_runs_when_the_flag_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RBCS_EXPERIMENTAL_SDE", "1")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert "Stochastic (experimental)" in [t.label for t in at.tabs]
    assert any(s.label == "Noise σ" for s in at.slider)
