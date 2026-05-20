#!/usr/bin/env python3
"""
Open-loop JEPA latent rollout vs oracle CLS encodings on ClimateBench NetCDF.

Uses the same uint8 + ImageNet preprocessing as ``evaluate_climatebench`` and the
same action convention as ``climatebench_lewm.convert`` (CO2 at index min(i+1,n-1)),
with Z-score normalization from the training HDF5 ``action`` column (matches LeWM
``get_column_normalizer``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
import xarray as xr
from stable_pretraining.data import dataset_stats
from torchvision.transforms import v2 as transforms_v2

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "external" / "le-wm") not in sys.path:
    sys.path.insert(0, str(_REPO / "external" / "le-wm"))

from climatebench_lewm.convert import (  # noqa: E402
    ConversionStats,
    _co2_forcing_values,
    _float_to_uint8,
    _prepare_tas_field,
)

def _pixel_transform(img_size: int):
    imagenet = dataset_stats.ImageNet
    return transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.Resize((img_size, img_size), antialias=True),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(**imagenet),
        ]
    )


def _action_zstats(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        act = np.asarray(f["action"][:], dtype=np.float64)
    act = act[~np.isnan(act).any(axis=1)]
    mean = np.mean(act, axis=0, keepdims=True)
    std = np.std(act, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _raw_action_timestep(co2_aligned: np.ndarray, i: int) -> np.ndarray:
    """Same indexing as HDF5 timestep ``i``: uses CO2[j] / co2_max, j=min(i+1,n-1)."""
    n = len(co2_aligned)
    j = min(i + 1, n - 1)
    return np.array([float(co2_aligned[j])], dtype=np.float32)


def _load_aligned_series(
    climatebench_root: Path,
    experiment: str,
) -> tuple[np.ndarray, xr.DataArray, int]:
    in_path = climatebench_root / f"inputs_{experiment}.nc"
    out_path = climatebench_root / f"outputs_{experiment}.nc"
    inds = xr.open_dataset(in_path)
    ods = xr.open_dataset(out_path)
    try:
        ti, to = int(inds.sizes["time"]), int(ods.sizes["time"])
        n = min(ti, to)
        co2 = _co2_forcing_values(inds)[:n].astype(np.float32)
        tas = ods["tas"]
        if "member" in tas.dims:
            tas = tas.mean(dim="member")
        tas = tas.isel(time=slice(0, n))
        return co2, tas, int(n)
    finally:
        inds.close()
        ods.close()


def _encode_cls(
    model: torch.nn.Module,
    ta: xr.DataArray,
    img_size: int,
    stats: ConversionStats,
    transform,
    device: torch.device,
) -> torch.Tensor:
    g = _prepare_tas_field(ta, img_size)
    u8 = _float_to_uint8(g, stats)
    hwc = np.stack([u8, u8, u8], axis=-1)
    pix = torch.from_numpy(hwc).permute(2, 0, 1)
    x = transform(pix).unsqueeze(0).unsqueeze(0).to(device)
    info: dict = {"pixels": x}
    out = model.encode(info)
    return out["emb"][0, 0].detach()


def save_latent_rollout_gif(
    horizons: list[int],
    mean_roll: list[float],
    mean_mov: list[float],
    title: str,
    gif_path: Path,
    duration_s: float = 6.0,
) -> None:
    h_max = len(horizons)
    interval_ms = max(120, int(1000 * duration_s / max(h_max, 1)))
    ymax = float(max(max(mean_roll), max(mean_mov)) * 1.15)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    (ln_r,) = ax.plot([], [], "o-", color="#1f77b4", lw=2, markersize=5)
    (ln_m,) = ax.plot([], [], "s-", color="#ff7f0e", lw=2, markersize=4, alpha=0.9)

    ax.set_xlim(min(horizons) - 0.5, max(horizons) + 0.5)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Forecast horizon (years)")
    ax.set_ylabel("Mean L2 in CLS latent space")
    ax.set_title(title, fontsize=10)
    ax.legend(
        [ln_r, ln_m],
        ["Open-loop rollout drift", "Oracle latent motion"],
        loc="upper left",
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    def update(k: int) -> tuple:
        kk = max(1, k + 1)
        xh = horizons[:kk]
        ln_r.set_data(xh, mean_roll[:kk])
        ln_m.set_data(xh, mean_mov[:kk])
        return ln_r, ln_m

    ani = FuncAnimation(
        fig, update, frames=h_max, interval=interval_ms, blit=False, repeat=True
    )
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(gif_path), writer="pillow")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--climatebench-root", type=Path, required=True)
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--lewm-object", type=Path, required=True)
    p.add_argument(
        "--stats-h5",
        type=Path,
        default=None,
        help="HDF5 used to match action Z-score stats (default: outputs/zenodo_stablewm/climatebench_train.h5)",
    )
    p.add_argument("--year-min", type=int, required=True)
    p.add_argument("--year-max", type=int, required=True)
    p.add_argument("--h-max", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/latent_rollout"),
    )
    args = p.parse_args()
    stats_h5 = args.stats_h5 or (_REPO / "outputs/zenodo_stablewm/climatebench_train.h5")
    if not stats_h5.is_file():
        sys.exit(f"Missing HDF5 for action stats: {stats_h5}")

    meta = json.loads(Path(args.meta).read_text())
    img_size = int(meta["img_size"])
    co2_max = float(meta.get("co2_max", 9500.0))
    conv_stats = ConversionStats(tas_min=float(meta["tas_min"]), tas_max=float(meta["tas_max"]))
    transform = _pixel_transform(img_size)
    a_mean, a_std = _action_zstats(stats_h5)

    co2_aligned, tas_full, n = _load_aligned_series(args.climatebench_root, args.experiment)

    tas = tas_full.sel(time=slice(args.year_min, args.year_max))
    if tas.sizes.get("time", 0) == 0:
        sys.exit(f"No tas in [{args.year_min}, {args.year_max}]")

    years = tas["time"].values.astype(np.int64)
    # Positional indices into co2_aligned/tas_full for each year (global index)
    year_to_idx = {}
    all_years = tas_full["time"].values.astype(np.int64)
    for i in range(len(all_years)):
        year_to_idx[int(all_years[i])] = i

    idx_list = sorted(year_to_idx[y] for y in years.astype(int))

    device = torch.device(args.device)
    model = torch.load(args.lewm_object, map_location=device, weights_only=False)
    model.eval()

    # Precompute oracle CLS for all global indices 0..n-1 needed
    oracle: list[torch.Tensor] = []
    with torch.no_grad():
        for gi in range(n):
            ta = tas_full.isel(time=gi)
            oracle.append(_encode_cls(model, ta, img_size, conv_stats, transform, device))

    h_max = min(args.h_max, n - 1)
    rollout_err: list[list[float]] = [[] for _ in range(h_max)]
    oracle_motion: list[list[float]] = [[] for _ in range(h_max)]

    starts = []
    for p_idx in idx_list:
        if p_idx + h_max >= n:
            continue
        starts.append(p_idx)

    if not starts:
        sys.exit("No valid start indices (check --h-max vs year span)")

    with torch.no_grad():
        for p in starts:
            hat = oracle[p].clone()
            for h in range(1, h_max + 1):
                i_from = p + h - 1
                raw = _raw_action_timestep(co2_aligned, i_from) / co2_max
                raw = raw.reshape(1, -1)
                normed = ((raw.astype(np.float64) - a_mean) / a_std).astype(
                    np.float32
                )
                act_t = torch.from_numpy(normed).to(device).unsqueeze(0)  # (1,1,A)
                act_emb = model.action_encoder(act_t)
                emb_ctx = hat.unsqueeze(0).unsqueeze(0)
                pred = model.predict(emb_ctx, act_emb)
                hat = pred[0, 0]
                tgt = oracle[p + h]
                rollout_err[h - 1].append(float(torch.linalg.norm(hat - tgt).cpu()))
                oracle_motion[h - 1].append(
                    float(torch.linalg.norm(tgt - oracle[p]).cpu())
                )

    mean_roll = [float(np.mean(rollout_err[h])) for h in range(h_max)]
    std_roll = [float(np.std(rollout_err[h])) for h in range(h_max)]
    mean_mov = [float(np.mean(oracle_motion[h])) for h in range(h_max)]
    std_mov = [float(np.std(oracle_motion[h])) for h in range(h_max)]
    horizons = list(range(1, h_max + 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_payload = {
        "experiment": args.experiment,
        "climatebench_root": str(args.climatebench_root.resolve()),
        "year_min": args.year_min,
        "year_max": args.year_max,
        "h_max": h_max,
        "n_starts": len(starts),
        "horizons": horizons,
        "mean_rollout_l2": mean_roll,
        "std_rollout_l2": std_roll,
        "mean_oracle_motion_l2": mean_mov,
        "std_oracle_motion_l2": std_mov,
    }
    stats_path = args.out_dir / "horizon_stats.json"
    stats_path.write_text(json.dumps(stats_payload, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(horizons, mean_roll, "o-", color="#1f77b4", label="Open-loop rollout L2 drift")
    ax.fill_between(
        horizons,
        [m - s for m, s in zip(mean_roll, std_roll)],
        [m + s for m, s in zip(mean_roll, std_roll)],
        alpha=0.2,
        color="#1f77b4",
    )
    ax.plot(horizons, mean_mov, "s-", color="#ff7f0e", alpha=0.85, label="Oracle latent motion (encode(t+h)-encode(t))")
    ax.fill_between(
        horizons,
        [m - s for m, s in zip(mean_mov, std_mov)],
        [m + s for m, s in zip(mean_mov, std_mov)],
        alpha=0.15,
        color="#ff7f0e",
    )
    ax.set_xlabel("Forecast horizon (years)")
    ax.set_ylabel("Mean L2 in CLS latent space")
    ttl = (
        f"Latent rollout vs oracle — {args.experiment}"
        f" ({args.year_min}–{args.year_max}); n_starts={len(starts)}, h≤{h_max}"
    )
    ax.set_title(ttl, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = args.out_dir / "latent_drift_horizon.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    gif_path = args.out_dir / "latent_drift_horizon.gif"
    save_latent_rollout_gif(horizons, mean_roll, mean_mov, ttl, gif_path)
    print(stats_path)
    print(png_path)
    print(gif_path)


if __name__ == "__main__":
    main()
