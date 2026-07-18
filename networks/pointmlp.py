"""Self-contained pointMLP models for classification and segmentation.

The implementation preserves pointMLP's hierarchical local grouping,
geometric affine normalization, residual pre-extraction, and residual
post-extraction MLPs.  FPS/KNN/interpolation are written in standard PyTorch
instead of relying on ``pointnet2_ops``.

Paper: https://arxiv.org/abs/2202.07123
Reference code: https://github.com/ma-xu/pointMLP-pytorch
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
    query_xyz: torch.Tensor,
    support_xyz: torch.Tensor,
    k: int,
    chunk_size: int = 256,
) -> torch.Tensor:
    k = min(max(int(k), 1), support_xyz.shape[1])
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        support = support_xyz.detach().float()
        for start in range(0, query_xyz.shape[1], chunk_size):
            query = query_xyz[:, start : start + chunk_size].detach().float()
            distances = torch.cdist(query, support)
            chunks.append(distances.topk(k, dim=-1, largest=False).indices)
    return torch.cat(chunks, dim=1)


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


class ResidualMLP(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.layers = nn.Sequential(
            nn.Linear(channels, expansion * channels),
            nn.GELU(),
            nn.Linear(expansion * channels, channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layers(self.norm(features))


class PointMLPStage(nn.Module):
    """LocalGrouper + pre-extraction + max pool + post-extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_size: int,
        reduction: int = 2,
        pre_blocks: int = 2,
        post_blocks: int = 2,
    ):
        super().__init__()
        self.group_size = int(group_size)
        self.reduction = int(reduction)
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, in_channels))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, in_channels))
        self.pre_projection = nn.Sequential(
            nn.Linear(2 * in_channels + 3, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
        self.pre_blocks = nn.ModuleList(
            ResidualMLP(out_channels) for _ in range(pre_blocks)
        )
        self.post_blocks = nn.ModuleList(
            ResidualMLP(out_channels) for _ in range(post_blocks)
        )

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = max(1, math.ceil(xyz.shape[1] / self.reduction))
        center_indices = _farthest_point_sample(xyz, count)
        center_xyz = _index_points(xyz, center_indices)
        center_features = _index_points(features, center_indices)
        neighbor_indices = _knn_indices(center_xyz, xyz, self.group_size)
        grouped_xyz = _index_points(xyz, neighbor_indices)
        grouped_features = _index_points(features, neighbor_indices)

        centered = grouped_features - center_features.unsqueeze(2)
        standard_deviation = centered.float().flatten(1).std(
            dim=1, keepdim=True, unbiased=False
        ).view(features.shape[0], 1, 1, 1)
        normalized = centered / standard_deviation.clamp_min(1e-5).to(centered.dtype)
        normalized = normalized * self.affine_alpha + self.affine_beta
        repeated_center = center_features.unsqueeze(2).expand(
            -1, -1, grouped_features.shape[2], -1
        )
        relative_xyz = grouped_xyz - center_xyz.unsqueeze(2)
        grouped = self.pre_projection(
            torch.cat([normalized, repeated_center, relative_xyz], dim=-1)
        )
        for block in self.pre_blocks:
            grouped = block(grouped)
        features = grouped.amax(dim=2)
        for block in self.post_blocks:
            features = block(features)
        return center_xyz, features


class PointMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, group_size: int = 24):
        super().__init__()
        if group_size <= 0:
            raise ValueError("pointmlp_k must be positive")
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        channels = (64, 128, 256, 512, 768)
        self.stages = nn.ModuleList(
            PointMLPStage(
                channels[index],
                channels[index + 1],
                group_size,
                reduction=2,
                pre_blocks=1 if index == 0 else 2,
                post_blocks=2,
            )
            for index in range(4)
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
            nn.GELU(),
            ResidualMLP(out_channels),
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


class PointMLPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_constraints: bool = False,
        constraint_dim: int = 15,
        group_size: int = 24,
    ):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        self.constraint_dim = int(constraint_dim)
        input_dim = 3 + (self.constraint_dim if self.use_constraints else 0)
        self.encoder = PointMLPEncoder(input_dim, group_size)
        self.head = nn.Sequential(
            nn.Linear(1536, 512),
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


class PointMLPSegmenter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_constraints: bool = False,
        constraint_dim: int = 15,
        group_size: int = 24,
    ):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        self.constraint_dim = int(constraint_dim)
        input_dim = 3 + (self.constraint_dim if self.use_constraints else 0)
        self.encoder = PointMLPEncoder(input_dim, group_size)
        self.propagation4 = FeaturePropagation(512, 768, 512)
        self.propagation3 = FeaturePropagation(256, 512, 256)
        self.propagation2 = FeaturePropagation(128, 256, 128)
        self.propagation1 = FeaturePropagation(64, 128, 64)
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
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
        features3 = self.propagation4(
            xyz_levels[3], xyz_levels[4], feature_levels[3], feature_levels[4]
        )
        features2 = self.propagation3(
            xyz_levels[2], xyz_levels[3], feature_levels[2], features3
        )
        features1 = self.propagation2(
            xyz_levels[1], xyz_levels[2], feature_levels[1], features2
        )
        features0 = self.propagation1(
            xyz_levels[0], xyz_levels[1], feature_levels[0], features1
        )
        return self.head(features0)


__all__ = ["PointMLPClassifier", "PointMLPSegmenter"]
