"""Minimal HDF5 that matches SWM reader (no ClimateBench paths required)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("stable_worldmodel")
from stable_worldmodel.data import HDF5Dataset


def test_hdf5_dataset_loads_pixels_action_schema():
    pytest.importorskip("h5py")
    import h5py
    import hdf5plugin  # noqa: F401

    h_img, w_img = 32, 32
    n_ep = 2
    lens = [5, 7]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        h5_path = root / "schema_test.h5"
        total = sum(lens)
        pixels = np.random.randint(
            0, 256, size=(total, h_img, w_img, 3), dtype=np.uint8
        )
        action = np.random.randn(total, 2).astype(np.float32)
        ep_len = np.array(lens, dtype=np.int32)
        ep_off = np.array([0, lens[0]], dtype=np.int64)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("pixels", data=pixels, compression=None)
            f.create_dataset("action", data=action, compression=None)
            f.create_dataset("ep_len", data=ep_len)
            f.create_dataset("ep_offset", data=ep_off)

        ds = HDF5Dataset(
            name="schema_test",
            cache_dir=str(root),
            frameskip=1,
            num_steps=3,
            keys_to_load=["pixels", "action"],
        )

        assert len(ds) > 0
        sample = ds[0]
        assert "pixels" in sample and "action" in sample
        pix = sample["pixels"]
        act = sample["action"]
        assert pix.ndim == 4
        assert pix.shape[-3] == 3
        assert act.ndim == 2
        assert torch.isfinite(pix).all()
        assert torch.isfinite(act).all()
