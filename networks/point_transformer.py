"""Pure-PyTorch Point Transformer models for classification and segmentation.

This is a self-contained adaptation of the vector-attention layer from
"Point Transformer" (Zhao et al., ICCV 2021).  It intentionally avoids the
original repository's custom ``pointops`` CUDA extension so it can use the
same environment and Stage 2 API as the other baselines in this project.

Paper: https://arxiv.org/abs/2012.09164
Reference code: https://github.com/POSTECH-CVLab/point-transformer
"""

from __future__ import annotations

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


def _knn_indices(xyz: torch.Tensor, k: int, chunk_size: int = 256) -> torch.Tensor:
    """Return full-resolution KNN indices without materializing BxNxN."""
    if k <= 0:
        raise ValueError("pointtransformer_k must be positive")
    k = min(k, xyz.shape[1])
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        support = xyz.detach().float()
        for start in range(0, xyz.shape[1], chunk_size):
            query = support[:, start : start + chunk_size]
            distances = torch.cdist(query, support)
            chunks.append(distances.topk(k, dim=-1, largest=False).indices)
    return torch.cat(chunks, dim=1)


class PointTransformerBlock(nn.Module):
    """Local vector self-attention with learned relative-position encoding."""

    def __init__(self, channels: int, k: int):
        super().__init__()
        self.k = int(k)
        self.norm1 = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.position = nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.attention = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.norm1(features)
        if indices is None:
            indices = _knn_indices(xyz, self.k)
        neighbor_xyz = _index_points(xyz, indices)
        neighbor_features = _index_points(normalized, indices)
        relative_position = xyz.unsqueeze(2) - neighbor_xyz
        position = self.position(relative_position)

        query = self.query(normalized).unsqueeze(2)
        key = self.key(neighbor_features)
        value = self.value(neighbor_features)
        weights = F.softmax(self.attention(query - key + position), dim=2)
        aggregated = (weights * (value + position)).sum(dim=2)
        features = features + self.projection(aggregated)
        return features + self.ffn(self.norm2(features))


class PointTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int = 64,
        depth: int = 3,
        k: int = 16,
    ):
        super().__init__()
        if width <= 0 or depth <= 0 or k <= 0:
            raise ValueError("Point Transformer width, depth, and k must be positive")
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            PointTransformerBlock(width, k) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        features = self.stem(points)
        indices = _knn_indices(xyz, self.blocks[0].k)
        for block in self.blocks:
            features = block(xyz, features, indices)
        return self.norm(features)


class PointTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_constraints: bool = False,
        constraint_dim: int = 15,
        k: int = 16,
        width: int = 64,
        depth: int = 3,
    ):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        self.constraint_dim = int(constraint_dim)
        input_dim = 3 + (self.constraint_dim if self.use_constraints else 0)
        self.encoder = PointTransformerEncoder(input_dim, width, depth, k)
        self.head = nn.Sequential(
            nn.Linear(2 * width, 2 * width),
            nn.LayerNorm(2 * width),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2 * width, num_classes),
        )

    def forward(
        self, xyz: torch.Tensor, constraints: torch.Tensor | None = None
    ) -> torch.Tensor:
        points = _validate_inputs(
            xyz, constraints, self.use_constraints, self.constraint_dim
        )
        features = self.encoder(xyz, points)
        pooled = torch.cat([features.amax(dim=1), features.mean(dim=1)], dim=-1)
        return F.log_softmax(self.head(pooled), dim=-1)


class PointTransformerSegmenter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_constraints: bool = False,
        constraint_dim: int = 15,
        k: int = 16,
        width: int = 64,
        depth: int = 3,
    ):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        self.constraint_dim = int(constraint_dim)
        input_dim = 3 + (self.constraint_dim if self.use_constraints else 0)
        self.encoder = PointTransformerEncoder(input_dim, width, depth, k)
        self.head = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.LayerNorm(2 * width),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * width, num_classes),
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
        features = self.encoder(xyz, points)
        global_features = torch.cat(
            [features.amax(dim=1), features.mean(dim=1)], dim=-1
        )
        global_features = global_features.unsqueeze(1).expand(-1, xyz.shape[1], -1)
        return self.head(torch.cat([features, global_features], dim=-1))


__all__ = [
    "PointTransformerClassifier",
    "PointTransformerSegmenter",
]
