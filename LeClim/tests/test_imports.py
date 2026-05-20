import pytest

pytest.importorskip("climatebench_lewm.convert")


def test_import_package():
    import climatebench_lewm  # noqa: F401

    assert hasattr(climatebench_lewm, "write_climatebench_hdf5")
