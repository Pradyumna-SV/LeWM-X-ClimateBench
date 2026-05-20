"""MLP + conv head: ViT patch tokens -> gridded ``tas`` (Kelvin)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TasDecoder(nn.Module):
    """
    Map encoder patch tokens (no CLS) to a dense ``img_size`` temperature field.

    Tokens are arranged on a ``token_grid``×``token_grid`` lattice (ViT with square patches).
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        token_grid: int,
        out_size: int,
        mid_channels: int = 128,
    ) -> None:
        super().__init__()
        self.token_grid = token_grid
        self.out_size = out_size
        self.in_proj = nn.Sequential(
            nn.Linear(hidden_dim, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, mid_channels),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_tokens :
            ``(batch, n_patches, hidden_dim)`` without CLS — ``n_patches`` must be a square.
        """
        b, p, d = patch_tokens.shape
        g = int(math.sqrt(p))
        if g * g != p:
            raise ValueError(f"Expected square patch count, got {p}")

        x = self.in_proj(patch_tokens)  # (B, P, mid)
        x = x.transpose(1, 2).reshape(b, -1, g, g)
        x = F.interpolate(
            x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )
        return self.out_conv(x).squeeze(1)


def patch_tokens_from_jepa_encoder(
    encoder: nn.Module, pixels_bt: torch.Tensor
) -> torch.Tensor:
    """Run frozen ViT encoder; return patch tokens ``(B, n_patches, D)`` (no CLS)."""
    out = encoder(pixels_bt, interpolate_pos_encoding=True)
    # HF ViT: last_hidden_state (B, 1 + n_patches, D)
    return out.last_hidden_state[:, 1:, :]
