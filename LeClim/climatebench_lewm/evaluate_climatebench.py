"""Evaluate ``tas`` predictions vs ClimateBench NetCDF (cos(lat) RMSE + NRMSE decomposition)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import zoom
from stable_pretraining.data import dataset_stats
from torchvision.transforms import v2 as transforms_v2

from climatebench_lewm.convert import ConversionStats, _float_to_uint8, _prepare_tas_field
from climatebench_lewm.decoder import TasDecoder, patch_tokens_from_jepa_encoder

logger = logging.getLogger(__name__)


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


def _cos_weights_sum(lat: np.ndarray, nlon: int) -> tuple[np.ndarray, float]:
    w1d = np.cos(np.deg2rad(lat)).astype(np.float64)
    w2 = w1d[:, np.newaxis]
    sum_w = float(w1d.sum() * nlon) + 1e-12
    return w2, sum_w


def _coslat_rmse_native(pred: np.ndarray, truth: np.ndarray, lat: np.ndarray) -> float:
    """Scalar RMSE with cos(lat) weights; ``pred``/``truth`` shape (nlat, nlon)."""
    w2, sum_w = _cos_weights_sum(lat, truth.shape[1])
    err = (pred.astype(np.float64) - truth.astype(np.float64)) ** 2
    return float(np.sqrt(np.sum(err * w2) / sum_w))


def _nrmse_decomposition(
    pred_series: list[np.ndarray],
    truth_series: list[np.ndarray],
    lat: np.ndarray,
) -> dict[str, float]:
    """
    Per-time fields ``(nlat, nlon)``. NRMSE normalized by truth variability on the same domain.

    * **total** — mean over time of cos‑lat RMSE of full fields; ``nrmse_total`` uses
      cos‑weighted RMS of truth **concatenated** over space-time.
    * **global** — RMSE of spatially aggregated (cos‑mean) scalars over time vs ``std`` of truth scalars.
    * **spatial** — mean cos‑lat RMSE of per‑timestep **anomaly** fields (subtract spatial cos‑mean);
      normalized by cos‑weighted RMS of those truth anomalies over space-time.
    """
    w2, sum_w = _cos_weights_sum(lat, truth_series[0].shape[1])

    def spat_mean(f: np.ndarray) -> float:
        return float(np.sum(f.astype(np.float64) * w2) / sum_w)

    per_t_mean_sq = [
        float(np.sum(t.astype(np.float64) ** 2 * w2) / sum_w) for t in truth_series
    ]
    sigma_tot = float(np.sqrt(np.mean(per_t_mean_sq) + 1e-12))

    rmses = []
    g_pred = []
    g_truth = []
    s_cmp_pred = []
    s_cmp_truth = []
    for p, t in zip(pred_series, truth_series):
        rmses.append(_coslat_rmse_native(p, t, lat))
        mp, mt = spat_mean(p), spat_mean(t)
        g_pred.append(mp)
        g_truth.append(mt)
        s_cmp_pred.append(p - mp)
        s_cmp_truth.append(t - mt)
    total_rmse = float(np.mean(rmses))
    g_pred_a = np.array(g_pred)
    g_truth_a = np.array(g_truth)
    global_rmse = float(np.sqrt(np.mean((g_pred_a - g_truth_a) ** 2)))
    sigma_g = float(np.std(g_truth_a) + 1e-12)

    spat_rms = []
    for pp, tt in zip(s_cmp_pred, s_cmp_truth):
        spat_rms.append(_coslat_rmse_native(pp, tt, lat))
    spatial_rmse = float(np.mean(spat_rms))
    stack_t = np.stack(s_cmp_truth, axis=0).astype(np.float64)
    per_t_spat = np.sum(stack_t**2 * w2, axis=(1, 2)) / sum_w
    sigma_s = float(np.sqrt(np.mean(per_t_spat) + 1e-12))

    return {
        "rmse_total_k": total_rmse,
        "rmse_global_mean_k": global_rmse,
        "rmse_spatial_pattern_k": spatial_rmse,
        "nrmse_total": total_rmse / sigma_tot,
        "nrmse_global": global_rmse / sigma_g,
        "nrmse_spatial": spatial_rmse / sigma_s,
        "truth_sigma_tot_k": sigma_tot,
        "truth_sigma_global_k": sigma_g,
        "truth_sigma_spatial_k": sigma_s,
        "n_times": float(len(truth_series)),
    }


def _img_to_native(
    pred_img: np.ndarray, native_shape: tuple[int, int]
) -> np.ndarray:
    """Resize square model output to NorESM ``(nlat, nlon)`` with bilinear zoom."""
    z = (native_shape[0] / pred_img.shape[0], native_shape[1] / pred_img.shape[1])
    return zoom(pred_img.astype(np.float64), z, order=1, mode="nearest").astype(
        np.float32
    )


def run_netcdf_eval(
    *,
    climatebench_root: Path,
    experiment: str,
    meta: dict[str, Any],
    lewm_object: Path,
    decoder_bundle: Path,
    device: str,
    max_times: int | None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict[str, Any]:
    img_size = int(meta["img_size"])
    stats = ConversionStats(
        tas_min=float(meta["tas_min"]), tas_max=float(meta["tas_max"])
    )

    dec_path = decoder_bundle / "tas_decoder.pt"
    if not dec_path.is_file():
        raise FileNotFoundError(dec_path)
    bundle = torch.load(dec_path, map_location=device, weights_only=False)
    dec_sd = bundle["decoder"]

    jepa = torch.load(lewm_object, map_location=device, weights_only=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    hidden = jepa.encoder.config.hidden_size
    psize = int(jepa.encoder.config.patch_size)
    token_grid = img_size // psize
    dec = TasDecoder(
        hidden_dim=hidden, token_grid=token_grid, out_size=img_size
    ).to(device)
    dec.load_state_dict(dec_sd)
    dec.eval()

    transform = _pixel_transform(img_size)
    in_path = climatebench_root / f"inputs_{experiment}.nc"
    out_path = climatebench_root / f"outputs_{experiment}.nc"
    if not in_path.is_file() or not out_path.is_file():
        raise FileNotFoundError(f"Need {in_path.name} and {out_path.name}")

    ods = xr.open_dataset(out_path)
    tas = ods["tas"]
    if "member" in tas.dims:
        tas = tas.mean(dim="member")

    if (year_min is None) ^ (year_max is None):
        raise ValueError("year_min and year_max must both be set or both omitted")
    if year_min is not None and year_max is not None:
        tas = tas.sel(time=slice(year_min, year_max))
        if tas.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No time steps in [{year_min}, {year_max}] for {out_path.name}"
            )

    pred_native_list: list[np.ndarray] = []
    truth_native_list: list[np.ndarray] = []
    lat_ref: np.ndarray | None = None

    n_t = int(tas.sizes["time"])
    if max_times is not None:
        n_t = min(n_t, max_times)

    with torch.no_grad():
        for t in range(n_t):
            ta = tas.isel(time=t)
            if lat_ref is None:
                lat_ref = np.asarray(ta["lat"].values, dtype=np.float64)

            truth_native = np.asarray(ta.values, dtype=np.float32)
            g = _prepare_tas_field(ta, img_size)
            u8 = _float_to_uint8(g, stats)
            hwc = np.stack([u8, u8, u8], axis=-1)
            pix = torch.from_numpy(hwc).permute(2, 0, 1)
            x = transform(pix).unsqueeze(0).to(device)

            tok = patch_tokens_from_jepa_encoder(jepa.encoder, x)
            pred_img = dec(tok).squeeze(0).cpu().numpy().astype(np.float32)
            pred_native = _img_to_native(pred_img, truth_native.shape)

            pred_native_list.append(pred_native)
            truth_native_list.append(truth_native)

    ods.close()
    assert lat_ref is not None

    metrics = _nrmse_decomposition(pred_native_list, truth_native_list, lat_ref)
    metrics["experiment"] = experiment
    metrics["climatebench_root"] = str(climatebench_root.resolve())
    if year_min is not None:
        metrics["year_min"] = float(year_min)
        metrics["year_max"] = float(year_max)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climatebench-root", type=Path, required=True)
    parser.add_argument(
        "--meta",
        type=Path,
        required=True,
        help="*.meta.json from the same conversion run as training data",
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--lewm-object", type=Path, required=True)
    parser.add_argument("--decoder-bundle", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--max-times", type=int, default=None)
    parser.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="Inclusive calendar year (ClimateBench ``time`` is an int year); use with --year-max",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="Inclusive calendar year; use with --year-min",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    meta = json.loads(args.meta.read_text())

    res = run_netcdf_eval(
        climatebench_root=args.climatebench_root,
        experiment=args.experiment,
        meta=meta,
        lewm_object=args.lewm_object,
        decoder_bundle=args.decoder_bundle,
        device=args.device,
        max_times=args.max_times,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2))
    logger.info("Wrote %s", args.out_json)


if __name__ == "__main__":
    main()
