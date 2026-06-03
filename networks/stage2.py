from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from functional.constraints import CONSTRAINT_DIM, split_constraint_tensor
from networks import utils


class PointwiseMLP(nn.Module):
    def __init__(self, channels: tuple[int, ...], activate_last: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=False))
            if i < len(channels) - 2 or activate_last:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
                layers.append(nn.SiLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, C] -> [B, N, C_out]
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ConstraintComponentFusion(nn.Module):
    """
    Extract features from each constraint component and fuse them with attention.
    """

    def __init__(self, feature_dim: int = 96):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                "primitive_type": PointwiseMLP((3 + 5, feature_dim, feature_dim)),
                "direction": PointwiseMLP((3 + 3, feature_dim, feature_dim)),
                "dimension": PointwiseMLP((3 + 1, feature_dim, feature_dim)),
                "continuity": PointwiseMLP((3 + 3, feature_dim, feature_dim)),
                "location": PointwiseMLP((3 + 3, feature_dim, feature_dim)),
            }
        )
        hidden = max(16, feature_dim // 2)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, xyz: torch.Tensor, constraints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = split_constraint_tensor(constraints)
        features = []
        for name, component in parts.items():
            features.append(self.encoders[name](torch.cat([xyz, component], dim=-1)))
        stacked = torch.stack(features, dim=2)  # [B, N, 5, C]
        weights = torch.softmax(self.attention(stacked).squeeze(-1), dim=2)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        return fused, weights


def _sample_indices(xyz: torch.Tensor, n_center: int) -> torch.Tensor:
    bsz, n_points, _ = xyz.shape
    if n_center >= n_points:
        return torch.arange(n_points, device=xyz.device).view(1, n_points).repeat(bsz, 1)
    return utils.fps(xyz, n_center)


class SetAbstractionBlock(nn.Module):
    def __init__(self, n_center: int, n_near: int, channel_in: int, channel_out: int):
        super().__init__()
        self.n_center = n_center
        self.n_near = n_near
        self.local_mlp = utils.MLP(
            2,
            (channel_in + 3, channel_out, channel_out),
            dropout=0.0,
            final_proc=True,
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_points, _ = xyz.shape
        n_center = min(self.n_center, n_points)
        n_near = min(self.n_near, n_points)

        fps_idx = _sample_indices(xyz, n_center)
        center_xyz = utils.index_points(xyz, fps_idx)
        dists = torch.cdist(center_xyz, xyz)
        group_idx = dists.topk(k=n_near, dim=-1, largest=False).indices

        grouped_xyz = utils.index_points(xyz, group_idx)
        grouped_features = utils.index_points(features, group_idx)
        relative_xyz = grouped_xyz - center_xyz.unsqueeze(2)
        local = torch.cat([grouped_features, relative_xyz], dim=-1).permute(0, 3, 1, 2)
        local = self.local_mlp(local).max(dim=-1)[0].transpose(1, 2)
        return center_xyz, local


class ConstraintAwareEncoder(nn.Module):
    def __init__(self, feature_dim: int = 96, latent_dim: int = 512):
        super().__init__()
        self.sa1 = SetAbstractionBlock(512, 32, feature_dim, 160)
        self.sa2 = SetAbstractionBlock(128, 32, 160, 320)
        self.proj = nn.Sequential(
            nn.Linear(320, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, xyz: torch.Tensor, fused_constraints: torch.Tensor) -> torch.Tensor:
        xyz1, fea1 = self.sa1(xyz, fused_constraints)
        _, fea2 = self.sa2(xyz1, fea1)
        pooled = fea2.max(dim=1)[0]
        return self.proj(pooled)


class CstNetStage2Classifier(nn.Module):
    def __init__(self, n_classes: int, feature_dim: int = 96, latent_dim: int = 512):
        super().__init__()
        self.constraint_fusion = ConstraintComponentFusion(feature_dim)
        self.encoder = ConstraintAwareEncoder(feature_dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, xyz: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        if constraints.shape[-1] != CONSTRAINT_DIM:
            raise ValueError(f"expected constraints [B, N, {CONSTRAINT_DIM}], got {tuple(constraints.shape)}")
        fused, _ = self.constraint_fusion(xyz, constraints)
        latent = self.encoder(xyz, fused)
        return F.log_softmax(self.head(latent), dim=-1)


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)


class CstNetStage2Diffusion(nn.Module):
    """
    Constraint-aware diffusion denoiser.

    Input: noisy point cloud at timestep t plus per-point constraints from the
    frozen Stage 1 extractor. Output: predicted noise [B, N, 3].
    """

    def __init__(
        self,
        feature_dim: int = 96,
        latent_dim: int = 512,
        time_dim: int = 128,
    ):
        super().__init__()
        self.constraint_fusion = ConstraintComponentFusion(feature_dim)
        self.encoder = ConstraintAwareEncoder(feature_dim, latent_dim)
        self.time_embedding = SinusoidalTimestepEmbedding(time_dim)
        self.decoder = PointwiseMLP(
            (3 + feature_dim + latent_dim + time_dim, 384, 192, 3),
            activate_last=False,
        )

    def forward(self, noisy_xyz: torch.Tensor, constraints: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if constraints.shape[-1] != CONSTRAINT_DIM:
            raise ValueError(f"expected constraints [B, N, {CONSTRAINT_DIM}], got {tuple(constraints.shape)}")
        fused, _ = self.constraint_fusion(noisy_xyz, constraints)
        latent = self.encoder(noisy_xyz, fused)
        time = self.time_embedding(timesteps)
        n_points = noisy_xyz.shape[1]
        decoder_in = torch.cat(
            [
                noisy_xyz,
                fused,
                latent.unsqueeze(1).expand(-1, n_points, -1),
                time.unsqueeze(1).expand(-1, n_points, -1),
            ],
            dim=-1,
        )
        return self.decoder(decoder_in)
