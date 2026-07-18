"""Pure-PyTorch PointNeXt models for classification and segmentation.

The implementation follows PointNeXt's scaled PointNet++ design: residual
local aggregation, inverted bottleneck MLPs, hierarchical set abstraction,
and feature-propagation decoding.  Sampling and neighborhood operations use
ordinary PyTorch so the file has no OpenPoints/CUDA-extension dependency.

Paper and reference code: https://github.com/guochengqian/PointNeXt
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_inputs(
    xyz: torch.Tensor,
    constraints: torch.Tensor | None,
    use_constraints: bool,
    constraint_dim: int,
) -> torch.Tensor:
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"expected xyz [B, N, 3], got {tuple(xyz.shape)}")
    if not use_constraints:
        return xyz
    expected = (xyz.shape[0], xyz.shape[1], constraint_dim)
    if constraints is None or tuple(constraints.shape) != expected:
        actual = None if constraints is None else tuple(constraints.shape)
        raise ValueError(
            f"constraint-enabled baseline expects constraints {expected}, got {actual}"
        )
    return torch.cat([xyz, constraints], dim=-1)


def _index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.view(points.shape[0], *([1] * (indices.ndim - 1)))
    return points[batch, indices]


def _farthest_point_sample(xyz: torch.Tensor, count: int) -> torch.Tensor:
    count = min(max(int(count), 1), xyz.shape[1])
    with torch.no_grad():
        work_xyz = xyz.detach().float()
        batch_size, n_points, _ = work_xyz.shape
        indices = torch.zeros(batch_size, count, dtype=torch.long, device=xyz.device)
        distances = torch.full(
            (batch_size, n_points), float("inf"), device=xyz.device
        )
        farthest = work_xyz.square().sum(dim=-1).argmax(dim=1)
        batch = torch.arange(batch_size, device=xyz.device)
        for index in range(count):
            indices[:, index] = farthest
            centroid = work_xyz[batch, farthest].unsqueeze(1)
            squared_distance = (work_xyz - centroid).square().sum(dim=-1)
            distances = torch.minimum(distances, squared_distance)
            farthest = distances.argmax(dim=1)
    return indices


def _knn_indices(
    query_xyz: torch.Tensor, support_xyz: torch.Tensor, k: int
) -> torch.Tensor:
    k = min(max(int(k), 1), support_xyz.shape[1])
    with torch.no_grad():
        distances = torch.cdist(query_xyz.detach().float(), support_xyz.detach().float())
        return distances.topk(k, dim=-1, largest=False).indices


def _interpolate(
    target_xyz: torch.Tensor,
    source_xyz: torch.Tensor,
    source_features: torch.Tensor,
) -> torch.Tensor:
    if source_xyz.shape[1] == 1:
        return source_features.expand(-1, target_xyz.shape[1], -1)
    indices = _knn_indices(target_xyz, source_xyz, min(3, source_xyz.shape[1]))
    neighbor_xyz = _index_points(source_xyz, indices)
    neighbor_features = _index_points(source_features, indices)
    distances = (target_xyz.unsqueeze(2).float() - neighbor_xyz.float()).square().sum(-1)
    reciprocal = distances.clamp_min(1e-10).reciprocal()
    weights = reciprocal / reciprocal.sum(dim=-1, keepdim=True)
    return (neighbor_features.float() * weights.unsqueeze(-1)).sum(dim=2).to(
        source_features.dtype
    )


class InvertedResidualMLP(nn.Module):
    def __init__(self, channels: int, k: int, expansion: int = 4):
        super().__init__()
        self.k = int(k)
        self.local_aggregation = nn.Sequential(
            nn.Linear(channels + 3, channels),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, expansion * channels),
            nn.GELU(),
            nn.Linear(expansion * channels, channels),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        indices = _knn_indices(xyz, xyz, self.k)
        neighbor_xyz = _index_points(xyz, indices)
        neighbor_features = _index_points(features, indices)
        relative_xyz = neighbor_xyz - xyz.unsqueeze(2)
        local = self.local_aggregation(
            torch.cat([neighbor_features, relative_xyz], dim=-1)
        ).amax(dim=2)
        features = features + local
        return features + self.mlp(self.norm(features))


class PointNeXtStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        reduction: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        self.k = int(k)
        self.reduction = int(reduction)
        self.neighbor_projection = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        self.skip_projection = nn.Linear(in_channels, out_channels)
        self.output_norm = nn.LayerNorm(out_channels)
        self.blocks = nn.ModuleList(
            InvertedResidualMLP(out_channels, k) for _ in range(depth)
        )

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = max(1, math.ceil(xyz.shape[1] / self.reduction))
        center_indices = _farthest_point_sample(xyz, count)
        center_xyz = _index_points(xyz, center_indices)
        center_features = _index_points(features, center_indices)
        neighbor_indices = _knn_indices(center_xyz, xyz, self.k)
        neighbor_xyz = _index_points(xyz, neighbor_indices)
        neighbor_features = _index_points(features, neighbor_indices)
        relative_xyz = neighbor_xyz - center_xyz.unsqueeze(2)
        aggregated = self.neighbor_projection(
            torch.cat([neighbor_features, relative_xyz], dim=-1)
        ).amax(dim=2)
        features = self.output_norm(
            aggregated + self.skip_projection(center_features)
        )
        for block in self.blocks:
            features = block(center_xyz, features)
        return center_xyz, features


class PointNeXtEncoder(nn.Module):
    def __init__(self, input_dim: int, k: int = 24):
        super().__init__()
        if k <= 0:
            raise ValueError("pointnext_k must be positive")
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            (
                PointNeXtStage(64, 96, k, reduction=4, depth=1),
                PointNeXtStage(96, 192, k, reduction=4, depth=2),
                PointNeXtStage(192, 384, k, reduction=4, depth=2),
            )
        )

    def forward(
        self, xyz: torch.Tensor, points: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        xyz_levels = [xyz]
        feature_levels = [self.stem(points)]
        for stage in self.stages:
            next_xyz, next_features = stage(xyz_levels[-1], feature_levels[-1])
            xyz_levels.append(next_xyz)
            feature_levels.append(next_features)
        return xyz_levels, feature_levels


class FeaturePropagation(nn.Module):
    def __init__(self, skip_channels: int, coarse_channels: int, out_channels: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(skip_channels + coarse_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        fine_xyz: torch.Tensor,
        coarse_xyz: torch.Tensor,
        fine_features: torch.Tensor,
        coarse_features: torch.Tensor,
    ) -> torch.Tensor:
        interpolated = _interpolate(fine_xyz, coarse_xyz, coarse_features)
        return self.fusion(torch.cat([fine_features, interpolated], dim=-1))


class PointNeXtClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_constraints: bool = False,
        constraint_dim: int = 15,
        k: int = 24,
    ):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        self.constraint_dim = int(constraint_dim)
        input_dim = 3 + (self.constraint_dim if self.use_constraints else 0)
        self.encoder = PointNeXtEncoder(input_dim, k)
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(
        self, xyz: torch.Tensor, constraints: torch.Tensor | None = None
    ) -> torch.Tensor:
        points = _validate_inputs(
            xyz, constraints, self.use_constraints, self.constraint_dim
        )
        _, levels = self.encoder(xyz, points)
        features = levels[-1]
        pooled = torch.cat([features.amax(dim=1), features.mean(dim=1)], dim=-1)
        return F.log_softmax(self.head(pooled), dim=-1)


class PointNeXtSegmenter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_constraints: bool = False,
        constraint_dim: int = 15,
        k: int = 24,
    ):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        self.constraint_dim = int(constraint_dim)
        input_dim = 3 + (self.constraint_dim if self.use_constraints else 0)
        self.encoder = PointNeXtEncoder(input_dim, k)
        self.propagation3 = FeaturePropagation(192, 384, 192)
        self.propagation2 = FeaturePropagation(96, 192, 96)
        self.propagation1 = FeaturePropagation(64, 96, 64)
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor | None = None,
        constraint_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del constraint_masks
        points = _validate_inputs(
            xyz, constraints, self.use_constraints, self.constraint_dim
        )
        xyz_levels, feature_levels = self.encoder(xyz, points)
        features2 = self.propagation3(
            xyz_levels[2], xyz_levels[3], feature_levels[2], feature_levels[3]
        )
        features1 = self.propagation2(
            xyz_levels[1], xyz_levels[2], feature_levels[1], features2
        )
        features0 = self.propagation1(
            xyz_levels[0], xyz_levels[1], feature_levels[0], features1
        )
        return self.head(features0)


__all__ = ["PointNeXtClassifier", "PointNeXtSegmenter"]
