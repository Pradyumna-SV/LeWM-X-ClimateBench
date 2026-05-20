"""
Convert ClimateBench NetCDF pairs to stable-worldmodel HDF5 for LeWM.

HDF5 layout matches stable_worldmodel.data.dataset.HDF5Dataset:
  pixels: (total_steps, H, W, C) uint8
  action: (total_steps, A) float32  (CO2 / co2_max; optional CH4 / ch4_max)
  ep_len, ep_offset: int index per episode
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # registers Blosc filter
import numpy as np
import xarray as xr
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

DEFAULT_CO2_MAX = 9500.0
DEFAULT_CH4_MAX = 0.8


@dataclass
class ConversionStats:
    """Running min/max over tas (Kelvin) for uint8 encoding."""

    tas_min: float
    tas_max: float


def _list_experiments(root: Path) -> list[str]:
    names: list[str] = []
    for p in sorted(root.glob("inputs_*.nc")):
        stem = p.stem  # inputs_historical
        exp = stem[len("inputs_") :]
        out = root / f"outputs_{exp}.nc"
        if out.is_file():
            names.append(exp)
        else:
            logger.warning("Skipping %s: missing %s", p.name, out.name)
    return names


def _prepare_tas_field(
    tas_t: xr.DataArray, img_size: int, flip_lat: bool = True
) -> np.ndarray:
    """2D float tas (lat, lon) → (img_size, img_size) float."""
    arr = tas_t.values.astype(np.float64)
    if flip_lat and tas_t.dims[0] == "lat":
        lats = tas_t["lat"].values
        if len(lats) > 1 and float(lats[0]) < float(lats[-1]):
            arr = np.flipud(arr)
        elif len(lats) > 1 and float(lats[0]) > float(lats[-1]):
            pass
    h, w = arr.shape
    z = (img_size / h, img_size / w)
    out = zoom(arr, z, order=1, mode="nearest")
    return out.astype(np.float32)


def _float_to_uint8(grid: np.ndarray, stats: ConversionStats) -> np.ndarray:
    lo, hi = stats.tas_min, stats.tas_max
    if hi <= lo:
        hi = lo + 1.0
    scaled = (grid - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _co2_forcing_values(inds: xr.Dataset) -> np.ndarray:
    """ClimateBench ``inputs_*.nc`` usually expose ``CO2``; ``hist-aer`` uses ``CO4``."""
    for key in ("CO2", "CO4"):
        if key in inds:
            return inds[key].values.astype(np.float32)
    have = ", ".join(sorted(inds.data_vars))
    raise KeyError(f"No CO2/CO4 forcing in inputs NetCDF (have: {have})")


def _aligned_time_steps(experiment: str, inds: xr.Dataset, ods: xr.Dataset) -> int:
    """Use min(input time, output time) when Zenodo pairs disagree (log once)."""
    ti, to = int(inds.sizes["time"]), int(ods.sizes["time"])
    if ti != to:
        logger.warning(
            "%s: inputs time=%d vs outputs time=%d — using first %d aligned steps",
            experiment,
            ti,
            to,
            min(ti, to),
        )
    return min(ti, to)


def _compute_global_tas_stats(
    root: Path,
    experiments: list[str],
    img_size: int,
) -> ConversionStats:
    gmin = np.inf
    gmax = -np.inf
    for exp in experiments:
        in_path = root / f"inputs_{exp}.nc"
        out_path = root / f"outputs_{exp}.nc"
        inds = xr.open_dataset(in_path)
        ds = xr.open_dataset(out_path)
        try:
            n = _aligned_time_steps(exp, inds, ds)
            tas = ds["tas"]
            if "member" in tas.dims:
                tas = tas.mean(dim="member")
            for t in range(n):
                g = _prepare_tas_field(tas.isel(time=t), img_size)
                gmin = min(gmin, float(np.nanmin(g)))
                gmax = max(gmax, float(np.nanmax(g)))
        finally:
            inds.close()
            ds.close()
    if not np.isfinite(gmin) or not np.isfinite(gmax):
        raise ValueError("Could not compute tas stats (check NetCDF contents)")
    logger.info(
        "Global tas range for uint8 scaling: %.4f .. %.4f (NetCDF units)",
        gmin,
        gmax,
    )
    return ConversionStats(tas_min=gmin, tas_max=gmax)


def _ch4_forcing_values(inds: xr.Dataset) -> np.ndarray:
    if "CH4" not in inds:
        raise KeyError(
            "include_ch4 requested but no CH4 in inputs NetCDF "
            f"(have: {sorted(inds.data_vars)})"
        )
    return inds["CH4"].values.astype(np.float32)


def _iter_episode_arrays(
    root: Path,
    experiment: str,
    img_size: int,
    stats: ConversionStats,
    co2_max: float,
    *,
    include_ch4: bool,
    ch4_max: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Build pixels, action, and float ``tas`` (Kelvin on resized grid) per timestep."""
    in_path = root / f"inputs_{experiment}.nc"
    out_path = root / f"outputs_{experiment}.nc"
    inds = xr.open_dataset(in_path)
    ods = xr.open_dataset(out_path)
    try:
        n = _aligned_time_steps(experiment, inds, ods)
        co2 = _co2_forcing_values(inds)[:n]
        ch4 = _ch4_forcing_values(inds)[:n] if include_ch4 else None
        tas = ods["tas"]
        if "member" in tas.dims:
            tas = tas.mean(dim="member")
        pixels_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []
        tas_k_list: list[np.ndarray] = []
        for i in range(n):
            g = _prepare_tas_field(tas.isel(time=i), img_size)
            tas_k_list.append(g.astype(np.float32))
            u8 = _float_to_uint8(g, stats)
            hwc = np.stack([u8, u8, u8], axis=-1)
            pixels_list.append(hwc)
        for i in range(n):
            j = min(i + 1, n - 1)
            if include_ch4:
                assert ch4 is not None
                a = np.array(
                    [co2[j] / co2_max, ch4[j] / ch4_max],
                    dtype=np.float32,
                )
            else:
                a = np.array([co2[j] / co2_max], dtype=np.float32)
            action_list.append(a)
        return pixels_list, action_list, tas_k_list
    finally:
        inds.close()
        ods.close()


def _init_h5_from_sample(
    f: h5py.File,
    sample_pixels: np.ndarray,
    sample_action: np.ndarray,
    sample_tas_k: np.ndarray,
) -> None:
    ph, pw, pc = sample_pixels.shape
    ah = sample_action.shape[0]
    assert pc == 3
    assert sample_tas_k.shape == (ph, pw)

    for name, sample_data in [
        ("pixels", np.asarray(sample_pixels)),
        ("action", np.asarray(sample_action)),
        ("tas_k", np.asarray(sample_tas_k)),
    ]:
        shape = (0,) + sample_data.shape
        maxshape = (None,) + sample_data.shape
        comp = None
        if sample_data.ndim >= 2:
            comp = hdf5plugin.Blosc(
                cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
            )
        f.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=sample_data.dtype,
            chunks=(100,) + sample_data.shape,
            compression=comp,
        )

    # tas_k: float32 (H, W) — same index as pixels; Kelvin after resize
    f.create_dataset(
        "ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1000,)
    )
    f.create_dataset(
        "ep_len", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1000,)
    )


def _append_episode(
    f: h5py.File,
    pixels: list[np.ndarray],
    actions: list[np.ndarray],
    tas_k: list[np.ndarray],
    global_ptr: int,
) -> int:
    ep_len = len(pixels)
    if ep_len != len(actions) or ep_len != len(tas_k):
        raise ValueError("pixels/actions/tas_k length mismatch")
    if ep_len < 2:
        logger.warning("Skipping episode with length %d (< 2)", ep_len)
        return global_ptr

    p_block = np.stack(pixels, axis=0)
    a_block = np.stack(actions, axis=0)
    tk_block = np.stack(tas_k, axis=0)

    for key, block in [
        ("pixels", p_block),
        ("action", a_block),
        ("tas_k", tk_block),
    ]:
        ds = f[key]
        cur = ds.shape[0]
        ds.resize(cur + ep_len, axis=0)
        ds[cur:] = block

    mi = f["ep_offset"].shape[0]
    f["ep_offset"].resize(mi + 1, axis=0)
    f["ep_len"].resize(mi + 1, axis=0)
    f["ep_offset"][mi] = global_ptr
    f["ep_len"][mi] = ep_len

    return global_ptr + ep_len


def write_climatebench_hdf5(
    climatebench_root: str | Path,
    output_h5: str | Path,
    *,
    experiments: list[str] | None = None,
    img_size: int = 224,
    co2_max: float = DEFAULT_CO2_MAX,
    stats: ConversionStats | None = None,
    include_ch4: bool = False,
    ch4_max: float = DEFAULT_CH4_MAX,
    overwrite: bool = False,
) -> Path:
    """
    Write a single HDF5 file with one episode per experiment (contiguous years).

    Parameters
    ----------
    climatebench_root :
        Directory containing ``inputs_{exp}.nc`` and ``outputs_{exp}.nc``.
    output_h5 :
        Full path to the output ``.h5`` file.
    experiments :
        Subset of experiment ids; default = all paired inputs/outputs in root.
    img_size :
        Square side length for bilinear-resampled ``tas`` (before uint8).
    co2_max :
        Normalizer for CO2 (ClimateBench baselines use 9500 ppm).
    stats :
        Global ``tas`` min/max in Kelvin after resize; computed if None.
    overwrite :
        If True, replace existing file.
    include_ch4 :
        If True, append normalized CH₄ (`CH4` / ``ch4_max``) as a second action column.
    ch4_max :
        Divisor for methane (ClimateBench-style default ``0.8``).
    """
    root = Path(climatebench_root).expanduser().resolve()
    outp = Path(output_h5).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CLIMATEBENCH_ROOT not found: {root}")

    exps = experiments or _list_experiments(root)
    if not exps:
        raise FileNotFoundError(f"No paired inputs_*/outputs_* in {root}")

    if stats is None:
        stats = _compute_global_tas_stats(root, exps, img_size)

    if outp.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to clobber {outp} (pass overwrite=True)"
            )
        outp.unlink()

    outp.parent.mkdir(parents=True, exist_ok=True)
    global_ptr = 0
    initialized = False

    with h5py.File(outp, "w") as f:
        for exp in exps:
            px, ac, tk = _iter_episode_arrays(
                root,
                exp,
                img_size,
                stats,
                co2_max,
                include_ch4=include_ch4,
                ch4_max=ch4_max,
            )
            if not initialized:
                _init_h5_from_sample(f, px[0], ac[0], tk[0])
                initialized = True
            global_ptr = _append_episode(f, px, ac, tk, global_ptr)

    write_conversion_metadata(
        outp,
        stats=stats,
        experiments=exps,
        img_size=img_size,
        co2_max=co2_max,
        ch4_max=ch4_max,
        include_ch4=include_ch4,
    )

    logger.info(
        "Wrote %s (%d episodes, %d total steps)",
        outp,
        len(exps),
        global_ptr,
    )
    return outp


def write_conversion_metadata(
    output_h5: Path,
    *,
    stats: ConversionStats,
    experiments: list[str],
    img_size: int,
    co2_max: float,
    ch4_max: float = DEFAULT_CH4_MAX,
    include_ch4: bool = False,
) -> Path:
    """Write ``{stem}.meta.json`` next to the HDF5 for decoder / eval reproducibility."""
    meta_path = output_h5.with_suffix(".meta.json")
    action_dim = 2 if include_ch4 else 1
    payload = {
        "tas_min": stats.tas_min,
        "tas_max": stats.tas_max,
        "img_size": img_size,
        "co2_max": co2_max,
        "ch4_max": ch4_max,
        "include_ch4": include_ch4,
        "action_dim": action_dim,
        "experiments": experiments,
        "hdf5_datasets": ["pixels", "action", "tas_k", "ep_offset", "ep_len"],
        "tas_k_units": "Kelvin",
        "tas_k_description": "Annual mean tas bilinearly resized to img_size (same as uint8 pixels before scaling).",
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", meta_path)
    return meta_path


def default_output_path(dataset_stem: str) -> Path:
    """Resolve ``$STABLEWM_HOME/{stem}.h5`` (stable-worldmodel convention)."""
    cache = Path(
        os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel")
    ).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{dataset_stem}.h5"
