"""Self-contained PointMamba models with a pure-PyTorch selective scan.

The model keeps PointMamba's space-filling-curve serialization, local patch
tokens, and non-hierarchical state-space encoder.  The selective scan is
implemented in ordinary PyTorch instead of requiring ``mamba_ssm``,
``causal_conv1d``, KNN-CUDA, or custom point-cloud CUDA extensions.

Paper: https://arxiv.org/abs/2402.10739
Reference code: https://github.com/LMD0311/PointMamba
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
    chunk_size: int = 256,
) -> torch.Tensor:
    if source_xyz.shape[1] == 1:
        return source_features.expand(-1, target_xyz.shape[1], -1)
    outputs: list[torch.Tensor] = []
    for start in range(0, target_xyz.shape[1], chunk_size):
        target = target_xyz[:, start : start + chunk_size]
        indices = _knn_indices(target, source_xyz, min(3, source_xyz.shape[1]))
        neighbor_xyz = _index_points(source_xyz, indices)
        neighbor_features = _index_points(source_features, indices)
        distances = (target.unsqueeze(2).float() - neighbor_xyz.float()).square().sum(-1)
        reciprocal = distances.clamp_min(1e-10).reciprocal()
        weights = reciprocal / reciprocal.sum(dim=-1, keepdim=True)
        outputs.append(
            (neighbor_features.float() * weights.unsqueeze(-1)).sum(dim=2).to(
                source_features.dtype
            )
        )
    return torch.cat(outputs, dim=1)


def _morton_order(xyz: torch.Tensor, bits: int = 10) -> torch.Tensor:
    """Serialize centers with a 3D Morton (Z-order) space-filling curve."""
    with torch.no_grad():
        xyz_float = xyz.detach().float()
        minimum = xyz_float.amin(dim=1, keepdim=True)
        extent = (xyz_float.amax(dim=1, keepdim=True) - minimum).clamp_min(1e-6)
        quantized = (((xyz_float - minimum) / extent) * (2**bits - 1)).long()
        code = torch.zeros(quantized.shape[:2], dtype=torch.long, device=xyz.device)
        for bit in range(bits):
            code |= ((quantized[..., 0] >> bit) & 1) << (3 * bit)
            code |= ((quantized[..., 1] >> bit) & 1) << (3 * bit + 1)
            code |= ((quantized[..., 2] >> bit) & 1) << (3 * bit + 2)
        return code.argsort(dim=1)


class SelectiveStateSpaceBlock(nn.Module):
    """Mamba-style gated selective SSM using a differentiable Python scan."""

    def __init__(self, channels: int, state_dim: int = 8, expansion: int = 2):
        super().__init__()
        inner = channels * expansion
        rank = max(4, channels // 16)
        self.inner = inner
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(channels)
        self.in_projection = nn.Linear(channels, 2 * inner)
        self.depthwise_conv = nn.Conv1d(
            inner, inner, kernel_size=3, padding=1, groups=inner
        )
        self.parameter_projection = nn.Linear(inner, rank + 2 * state_dim, bias=False)
        self.delta_projection = nn.Linear(rank, inner)
        initial_a = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.a_log = nn.Parameter(initial_a.log().repeat(inner, 1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.out_projection = nn.Linear(inner, channels)

    def _scan(
        self,
        values: torch.Tensor,
        delta: torch.Tensor,
        input_state: torch.Tensor,
        output_state: torch.Tensor,
    ) -> torch.Tensor:
        values = values.float()
        delta = delta.float()
        input_state = input_state.float()
        output_state = output_state.float()
        a = -torch.exp(self.a_log.float())
        state = values.new_zeros(
            values.shape[0], self.inner, self.state_dim
        )
        outputs: list[torch.Tensor] = []
        for index in range(values.shape[1]):
            step = delta[:, index].unsqueeze(-1)
            transition = torch.exp(step * a.unsqueeze(0))
            state = transition * state + (
                step
                * input_state[:, index].unsqueeze(1)
                * values[:, index].unsqueeze(-1)
            )
            output = (
                state * output_state[:, index].unsqueeze(1)
            ).sum(dim=-1)
            output = output + self.skip.float() * values[:, index]
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        values, gate = self.in_projection(self.norm(features)).chunk(2, dim=-1)
        values = self.depthwise_conv(values.transpose(1, 2)).transpose(1, 2)
        values = F.silu(values)
        parameters = self.parameter_projection(values)
        rank = parameters.shape[-1] - 2 * self.state_dim
        delta_raw, input_state, output_state = torch.split(
            parameters, (rank, self.state_dim, self.state_dim), dim=-1
        )
        delta = F.softplus(self.delta_projection(delta_raw))
        forward = self._scan(values, delta, input_state, output_state)
        backward = self._scan(
            values.flip(1),
            delta.flip(1),
            input_state.flip(1),
            output_state.flip(1),
        ).flip(1)
        mixed = (0.5 * (forward + backward)).to(gate.dtype) * F.silu(gate)
        return residual + self.out_projection(mixed)


class PointMambaEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int = 64,
        depth: int = 2,
        num_tokens: int = 128,
        group_size: int = 32,
    ):
        super().__init__()
        if min(width, depth, num_tokens, group_size) <= 0:
            raise ValueError("PointMamba dimensions must be positive")
        self.num_tokens = int(num_tokens)
        self.group_size = int(group_size)
        self.point_stem = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.patch_embedding = nn.Sequential(
            nn.Linear(width + 3, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.position_embedding = nn.Sequential(
            nn.Linear(3, width), nn.GELU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            SelectiveStateSpaceBlock(width) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self, xyz: torch.Tensor, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_features = self.point_stem(points)
        center_indices = _farthest_point_sample(xyz, self.num_tokens)
        center_xyz = _index_points(xyz, center_indices)
        neighbor_indices = _knn_indices(center_xyz, xyz, self.group_size)
        grouped_xyz = _index_points(xyz, neighbor_indices)
        grouped_features = _index_points(point_features, neighbor_indices)
        relative_xyz = grouped_xyz - center_xyz.unsqueeze(2)
        tokens = self.patch_embedding(
            torch.cat([relative_xyz, grouped_features], dim=-1)
        ).amax(dim=2)

        order = _morton_order(center_xyz)
        center_xyz = _index_points(center_xyz, order)
        tokens = _index_points(tokens, order) + self.position_embedding(center_xyz)
        for block in self.blocks:
            tokens = block(tokens)
        return center_xyz, self.norm(tokens), point_features


class PointMambaClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_constraints: bool = False,
        constraint_dim: int = 15,
        num_tokens: int = 128,
        group_size: int = 32,
        width: int = 64,
        depth: int = 2,
    ):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        self.constraint_dim = int(constraint_dim)
        input_dim = 3 + (self.constraint_dim if self.use_constraints else 0)
        self.encoder = PointMambaEncoder(
            input_dim, width, depth, num_tokens, group_size
        )
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
        _, tokens, _ = self.encoder(xyz, points)
        pooled = torch.cat([tokens.amax(dim=1), tokens.mean(dim=1)], dim=-1)
        return F.log_softmax(self.head(pooled), dim=-1)


class PointMambaSegmenter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_constraints: bool = False,
        constraint_dim: int = 15,
        num_tokens: int = 128,
        group_size: int = 32,
        width: int = 64,
        depth: int = 2,
    ):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        self.constraint_dim = int(constraint_dim)
        input_dim = 3 + (self.constraint_dim if self.use_constraints else 0)
        self.encoder = PointMambaEncoder(
            input_dim, width, depth, num_tokens, group_size
        )
        self.head = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
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
        center_xyz, tokens, point_features = self.encoder(xyz, points)
        interpolated = _interpolate(xyz, center_xyz, tokens)
        global_features = torch.cat(
            [tokens.amax(dim=1), tokens.mean(dim=1)], dim=-1
        ).unsqueeze(1).expand(-1, xyz.shape[1], -1)
        return self.head(
            torch.cat([point_features, interpolated, global_features], dim=-1)
        )


__all__ = ["PointMambaClassifier", "PointMambaSegmenter"]
