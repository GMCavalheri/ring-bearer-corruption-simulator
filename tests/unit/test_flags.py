import pytest

from corruption.flags import FLAGS, is_enabled


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_flag_is_on_for_truthy_values(value: str) -> None:
    assert is_enabled("EXPERIMENTAL_SDE", {"RBCS_EXPERIMENTAL_SDE": value})


@pytest.mark.parametrize("value", ["0", "false", "", "maybe"])
def test_flag_is_off_for_other_values(value: str) -> None:
    assert not is_enabled("EXPERIMENTAL_SDE", {"RBCS_EXPERIMENTAL_SDE": value})


def test_flag_defaults_to_off_when_unset() -> None:
    assert not is_enabled("EXPERIMENTAL_SDE", {})
    assert all(not flag.default for flag in FLAGS.values())


def test_flag_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RBCS_EXPERIMENTAL_SDE", "1")
    assert is_enabled("EXPERIMENTAL_SDE")
    monkeypatch.delenv("RBCS_EXPERIMENTAL_SDE")
    assert not is_enabled("EXPERIMENTAL_SDE")


def test_unknown_flag_raises() -> None:
    with pytest.raises(KeyError):
        is_enabled("NOT_A_FLAG", {})
