"""Train a ``TasDecoder`` on top of a pickled LeWM ``JEPA`` object (frozen encoder)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from stable_pretraining.data import dataset_stats
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms_v2

from climatebench_lewm.decoder import TasDecoder, patch_tokens_from_jepa_encoder

logger = logging.getLogger(__name__)


def _build_pixel_transform(img_size: int):
    imagenet = dataset_stats.ImageNet
    return transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.Resize((img_size, img_size), antialias=True),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(**imagenet),
        ]
    )


class _H5TasSteps(Dataset):
    """One sample per HDF5 row: uint8 ``pixels`` + float ``tas_k``."""

    def __init__(self, h5_path: Path, img_size: int, indices: np.ndarray) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.transform = _build_pixel_transform(img_size)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> dict:
        idx = int(self.indices[i])
        with h5py.File(self.h5_path, "r") as f:
            pix = np.asarray(f["pixels"][idx])  # H,W,3 uint8
            tas = np.asarray(f["tas_k"][idx]).astype(np.float32)
        # HWC uint8 -> transform expects PIL or tensor; ToImage accepts uint8 CHW after numpy
        t = torch.from_numpy(pix).permute(2, 0, 1)
        x = self.transform(t)
        return {"pixels": x, "tas_k": torch.from_numpy(tas)}


def _episode_train_indices(
    ep_offset: np.ndarray, ep_len: np.ndarray, train_frac: float
) -> np.ndarray:
    """First ``train_frac`` episodes -> train; rest validation. Returns global step indices."""
    n_ep = len(ep_len)
    n_train = max(1, int(round(n_ep * train_frac)))
    train_idx: list[int] = []
    val_idx: list[int] = []
    for e in range(n_ep):
        start = int(ep_offset[e])
        length = int(ep_len[e])
        rows = np.arange(start, start + length, dtype=np.int64)
        if e < n_train:
            train_idx.append(rows)
        else:
            val_idx.append(rows)
    return np.concatenate(train_idx), np.concatenate(val_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path, help="HDF5 with pixels, tas_k")
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="*.meta.json from converter (defaults to h5 with .meta.json)",
    )
    parser.add_argument(
        "--lewm-object",
        required=True,
        type=Path,
        help="Pickled JEPA from ModelObjectCallBack (e.g. lewm_epoch_20_object.ckpt)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("decoder_bundle"),
        help="Directory: decoder.pt, config.json",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    meta_path = args.meta or args.h5.with_suffix(".meta.json")
    if not meta_path.is_file():
        sys.exit(f"Missing meta JSON {meta_path} (run climatebench-to-lewm-hdf5 to generate)")

    meta = json.loads(meta_path.read_text())
    img_size = int(meta["img_size"])

    with h5py.File(args.h5, "r") as f:
        ep_off = np.asarray(f["ep_offset"])
        ep_len = np.asarray(f["ep_len"])
        n_steps = f["pixels"].shape[0]

    train_rows, val_rows = _episode_train_indices(ep_off, ep_len, args.train_frac)
    train_ds = _H5TasSteps(args.h5, img_size, train_rows)
    val_ds = _H5TasSteps(args.h5, img_size, val_rows)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ld = torch.load(
        args.lewm_object, map_location=args.device, weights_only=False
    )
    if isinstance(ld, dict) and "state_dict" in ld:
        raise ValueError(
            "Expected JEPA object checkpoint from ModelObjectCallBack (torch.save(model)), "
            "not Lightning state_dict."
        )
    jepa = ld
    jepa = jepa.to(args.device)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    hidden = jepa.encoder.config.hidden_size
    patch = meta.get("img_size", img_size) // jepa.encoder.config.patch_size
    token_grid = int(patch)
    n_tok = token_grid * token_grid
    # sanity vs first forward
    dec = TasDecoder(hidden_dim=hidden, token_grid=token_grid, out_size=img_size).to(
        args.device
    )
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(args.epochs):
        dec.train()
        tot = 0.0
        n_bt = 0
        for batch in train_loader:
            pix = batch["pixels"].to(args.device)
            y = batch["tas_k"].to(args.device)
            with torch.no_grad():
                tok = patch_tokens_from_jepa_encoder(jepa.encoder, pix)
            pred = dec(tok)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            n_bt += 1
        tr = tot / max(1, n_bt)

        dec.eval()
        vtot = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                pix = batch["pixels"].to(args.device)
                y = batch["tas_k"].to(args.device)
                tok = patch_tokens_from_jepa_encoder(jepa.encoder, pix)
                pred = dec(tok)
                loss = F.mse_loss(pred, y)
                vtot += float(loss.item())
                vn += 1
        va = vtot / max(1, vn)
        logger.info("epoch %d train_mse %.6f val_mse %.6f", epoch, tr, va)
        if va < best_val:
            best_val = va
            torch.save(
                {"decoder": dec.state_dict(), "meta": meta},
                args.out / "tas_decoder.pt",
            )

    cfg = {
        "h5": str(args.h5.resolve()),
        "meta": str(meta_path.resolve()),
        "lewm_object": str(args.lewm_object.resolve()),
        "img_size": img_size,
        "encoder_hidden": hidden,
        "patch_size": int(jepa.encoder.config.patch_size),
        "token_grid": token_grid,
        "n_patch_tokens": n_tok,
        "train_frac": args.train_frac,
        "epochs": args.epochs,
        "best_val_mse_k2": best_val,
    }
    (args.out / "decoder_train_config.json").write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote %s", args.out / "tas_decoder.pt")


if __name__ == "__main__":
    main()
