#!/usr/bin/env python3
"""Plot train/val curves from Lightning CSVLogger metrics.csv (LeWM / stable-pretraining)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _to_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_series(metrics_csv: Path) -> dict[str, list[tuple[int, float]]]:
    """Return epoch-indexed validation metrics (last row wins per epoch)."""
    out: dict[str, dict[int, float]] = {
        "val_loss": {},
        "val_pred": {},
        "val_sigreg": {},
    }
    with metrics_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ep_raw = (row.get("epoch") or "").strip()
            if ep_raw == "":
                continue
            try:
                ep = int(float(ep_raw))
            except ValueError:
                continue
            if (v := _to_float(row.get("validate/loss_epoch", "") or "")) is not None:
                out["val_loss"][ep] = v
            if (v := _to_float(row.get("validate/pred_loss_epoch", "") or "")) is not None:
                out["val_pred"][ep] = v
            if (v := _to_float(row.get("validate/sigreg_loss_epoch", "") or "")) is not None:
                out["val_sigreg"][ep] = v

    def sorted_pairs(d: dict[int, float]) -> list[tuple[int, float]]:
        return sorted(d.items())

    return {k: sorted_pairs(v) for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics_csv", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("training_curves.png"))
    args = p.parse_args()

    series = load_series(args.metrics_csv)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for key, label, color in (
        ("val_loss", "validate / loss (epoch)", "C0"),
        ("val_pred", "validate / pred_loss (epoch)", "C1"),
        ("val_sigreg", "validate / sigreg_loss (epoch)", "C2"),
    ):
        pts = series[key]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=label, color=color, marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
