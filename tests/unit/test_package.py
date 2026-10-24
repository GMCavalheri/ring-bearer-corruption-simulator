import corruption


def test_package_exposes_version() -> None:
    assert corruption.__version__.count(".") == 2
