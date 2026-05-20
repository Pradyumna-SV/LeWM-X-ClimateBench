"""Integration tests requiring extracted ClimateBench ``train_val`` NetCDF files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("xarray")
import h5py
from climatebench_lewm.convert import write_climatebench_hdf5

pytest.importorskip("stable_worldmodel")
from stable_worldmodel.data import HDF5Dataset


def _root():
    r = os.environ.get("CLIMATEBENCH_ROOT")
    return Path(r).resolve() if r else None


@pytest.mark.skipif(
    _root() is None or not _root().is_dir(),
    reason="Set CLIMATEBENCH_ROOT to extracted ClimateBench train_val directory",
)
def test_convert_and_load_dataset_roundtrip():
    root = _root()
    assert root is not None

    paired = list(root.glob("inputs_*.nc"))
    if not paired:
        pytest.fail(f"No inputs_*.nc under {root}")

    with tempfile.TemporaryDirectory() as tmp:
        outp = Path(tmp) / "roundtrip.h5"
        write_climatebench_hdf5(
            root,
            outp,
            img_size=64,
            overwrite=True,
        )
        assert outp.is_file()
        assert outp.with_suffix(".meta.json").is_file()
        with h5py.File(outp, "r") as f:
            assert "tas_k" in f
            assert f["tas_k"].shape[1:] == (64, 64)

        ds = HDF5Dataset(
            name="roundtrip",
            cache_dir=str(Path(tmp)),
            frameskip=1,
            num_steps=2,
            keys_to_load=["pixels", "action"],
        )
        sample = ds[0]
        assert sample["pixels"].shape[0] == 2  # num_steps after frameskip on pixels
        assert torch.isfinite(sample["pixels"]).all()
        assert torch.isfinite(sample["action"]).all()
