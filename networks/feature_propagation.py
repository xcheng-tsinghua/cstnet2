from __future__ import annotations

import torch
import torch.nn as nn

from networks import utils


class FeaturePropagation(nn.Module):
    """PointNet++ three-neighbor inverse-distance feature propagation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("feature propagation channels must be positive")
        self.eps = float(eps)
        hidden_channels = int(hidden_channels or out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        xyz_fine: torch.Tensor,
        xyz_coarse: torch.Tensor,
        feat_fine: torch.Tensor | None,
        feat_coarse: torch.Tensor,
    ) -> torch.Tensor:
        if xyz_fine.ndim != 3 or xyz_fine.shape[-1] != 3:
            raise ValueError(f"expected xyz_fine [B, N, 3], got {tuple(xyz_fine.shape)}")
        if xyz_coarse.ndim != 3 or xyz_coarse.shape[-1] != 3:
            raise ValueError(f"expected xyz_coarse [B, S, 3], got {tuple(xyz_coarse.shape)}")
        if feat_coarse.ndim != 3 or feat_coarse.shape[:2] != xyz_coarse.shape[:2]:
            raise ValueError("feat_coarse must align with xyz_coarse")
        if xyz_fine.shape[0] != xyz_coarse.shape[0]:
            raise ValueError("fine and coarse batches must have the same size")
        if feat_fine is not None and (
            feat_fine.ndim != 3 or feat_fine.shape[:2] != xyz_fine.shape[:2]
        ):
            raise ValueError("feat_fine must align with xyz_fine")

        n_coarse = xyz_coarse.shape[1]
        if n_coarse == 0:
            raise ValueError("cannot propagate from an empty coarse point set")
        if n_coarse == 1:
            interpolated = feat_coarse.expand(-1, xyz_fine.shape[1], -1)
        else:
            # Compute the fine-to-coarse distances once, then reuse the same
            # top-k result for both indices and interpolation weights.
            distances = torch.cdist(xyz_fine, xyz_coarse)
            nearest = distances.topk(k=min(3, n_coarse), dim=-1, largest=False)
            inverse = nearest.values.clamp_min(self.eps).reciprocal()
            weights = inverse / inverse.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            neighbor_features = utils.index_points(feat_coarse, nearest.indices)
            interpolated = (neighbor_features * weights.unsqueeze(-1)).sum(dim=2)

        fused = interpolated if feat_fine is None else torch.cat([feat_fine, interpolated], dim=-1)
        return self.mlp(fused)

