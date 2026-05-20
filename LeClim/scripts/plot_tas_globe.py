#!/usr/bin/env python3
"""Plot global `tas` (or bias) on a map — Robinson projection + optional Natural Earth land."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as e:  # pragma: no cover
    ccrs = None
    cfeature = None
    _CARTOPY_ERR = e
else:
    _CARTOPY_ERR = None


def main() -> None:
    if _CARTOPY_ERR is not None:
        raise SystemExit(
            "cartopy is required for plot_tas_globe.py "
            f"(pip install -e '.[climate-viz]'). Original error: {_CARTOPY_ERR}"
        )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", required=True, help="Path to .npy 1D latitude (degrees)")
    p.add_argument("--lon", required=True, help="Path to .npy 1D longitude (degrees)")
    p.add_argument("--field", required=True, help="Path to .npy 2D field (nlat, nlon)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output .png")
    p.add_argument("--title", default="Surface air temperature (K)")
    p.add_argument("--cmap", default="viridis")
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
    )
    p.add_argument("--vmax", type=float, default=None)
    args = p.parse_args()

    lat = np.load(args.lat)
    lon = np.load(args.lon)
    fld = np.load(args.field)

    lon2 = np.where(lon > 180.0, lon - 360.0, lon)
    lon_ctr = float(np.mean(lon2))

    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=lon_ctr))
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.92", edgecolor="0.6", linewidth=0.3)
    ax.coastlines(resolution="50m", linewidth=0.5)

    vmin = args.vmin if args.vmin is not None else float(np.nanmin(fld))
    vmax = args.vmax if args.vmax is not None else float(np.nanmax(fld))

    mesh = ax.pcolormesh(
        lon2,
        lat,
        fld,
        transform=ccrs.PlateCarree(),
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    plt.colorbar(mesh, ax=ax, shrink=0.6, label="K")
    ax.set_title(args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
