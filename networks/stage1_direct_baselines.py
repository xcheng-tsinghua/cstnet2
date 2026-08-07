"""XYZ-only Stage 1 baselines that directly predict constraint components.

These models deliberately bypass primitive-instance embeddings, clustering,
and primitive fitting.  Every registered backbone produces a per-point feature
map which is consumed by four independent heads for primitive type, direction,
dimension, and location.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from functional.constraints import (
    INVALID_DIRECTION,
    N_PRIMITIVES,
    canonicalize_directions,
    constraints_to_tensor,
)
from networks.attn_3dgcn import Attn3DGcnPointEmbedding
from networks.dgcnn_gn import DGCNGn
from networks.point_mamba import PointMambaSegmenter
from networks.point_net import PointNetPointEmbedding
from networks.point_net2 import PointNet2PointEmbedding
from networks.point_transformer import PointTransformerSegmenter
from networks.pointmlp import PointMLPSegmenter
from networks.pointnext import PointNeXtSegmenter


DIRECT_BASELINE_MODEL_NAMES = (
    "pointnet",
    "pointnet2",
    "attn3dgcn",
    "dgcnn",
    "pointtransformer",
    "pointmamba",
    "pointnext",
    "pointmlp",
)

_MODEL_ALIASES = {
    "attn_3dgcn": "attn3dgcn",
    "pointnet++": "pointnet2",
    "point_transformer": "pointtransformer",
    "point_mamba": "pointmamba",
    "point_next": "pointnext",
    "point_mlp": "pointmlp",
}

_DEFAULT_CONFIG: dict[str, Any] = {
    "model": "pointnet2",
    "feature_dim": 128,
    "head_hidden_dim": 128,
    "head_dropout": 0.2,
    "dgcnn_k": 20,
    "attn_neighbors": 20,
    "attn_k": 16,
    "pointtransformer_k": 16,
    "pointtransformer_width": 64,
    "pointtransformer_depth": 3,
    "pointmamba_tokens": 128,
    "pointmamba_group_size": 32,
    "pointmamba_width": 64,
    "pointmamba_depth": 2,
    "pointnext_k": 24,
    "pointmlp_group_size": 24,
}


def _config_value(source: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def stage1_direct_model_config(source: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Resolve all architecture values needed to reconstruct a checkpoint."""
    model_name = str(
        _config_value(source, "model", _DEFAULT_CONFIG["model"])
        or _DEFAULT_CONFIG["model"]
    ).lower()
    model_name = _MODEL_ALIASES.get(model_name, model_name)
    if model_name not in DIRECT_BASELINE_MODEL_NAMES:
        raise ValueError(
            f"unknown Stage 1 direct baseline {model_name!r}; "
            f"expected one of {DIRECT_BASELINE_MODEL_NAMES}"
        )

    config: dict[str, Any] = {"model": model_name}
    for name, default in _DEFAULT_CONFIG.items():
        if name == "model":
            continue
        value = _config_value(source, name, default)
        config[name] = float(value) if name == "head_dropout" else int(value)

    positive_integer_fields = [
        name for name in config if name not in {"model", "head_dropout"}
    ]
    for name in positive_integer_fields:
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= config["head_dropout"] < 1.0:
        raise ValueError("head_dropout must be in [0, 1)")
    return config


def _without_final_layer(module: nn.Sequential) -> nn.Sequential:
    children = list(module.children())
    if len(children) < 2:
        raise ValueError("segmentation head must contain a removable final layer")
    return nn.Sequential(*children[:-1])


class _PointNetFeatureAdapter(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.output_dim = int(feature_dim)
        self.model = PointNetPointEmbedding(channel_out=self.output_dim)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        features = self.model(xyz.transpose(1, 2).contiguous())
        return features.transpose(1, 2).contiguous()


class _PointNet2FeatureAdapter(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        # The existing PointNet++ embedding constructor exposes channel_out for
        # API compatibility, but its feature-propagation decoder is fixed at
        # 128 channels.  Keep that native width explicit and let the common
        # projection below map it to the requested direct-baseline width.
        self.output_dim = 128
        self.model = PointNet2PointEmbedding(channel_out=int(feature_dim))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        features = self.model(xyz.transpose(1, 2).contiguous())
        return features.transpose(1, 2).contiguous()


class _Attn3DGCNFeatureAdapter(nn.Module):
    def __init__(self, feature_dim: int, n_neighbors: int, attn_k: int):
        super().__init__()
        self.output_dim = int(feature_dim)
        self.n_neighbors = int(n_neighbors)
        self.attn_k = int(attn_k)
        self.model = Attn3DGcnPointEmbedding(
            channel_coord=3,
            channel_fea=0,
            channel_out=self.output_dim,
            n_neighbor=self.n_neighbors,
            attn_k=self.attn_k,
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.shape[1] <= self.n_neighbors:
            raise ValueError(
                "Attention 3DGCN requires more points than attn_neighbors; "
                f"got N={xyz.shape[1]}, attn_neighbors={self.n_neighbors}"
            )
        if xyz.shape[1] < self.attn_k:
            raise ValueError(
                f"Attention 3DGCN requires at least attn_k={self.attn_k} points, "
                f"got {xyz.shape[1]}"
            )
        features = self.model(xyz.transpose(1, 2).contiguous())
        return features.transpose(1, 2).contiguous()


class _DGCNNFeatureAdapter(nn.Module):
    def __init__(self, feature_dim: int, k: int):
        super().__init__()
        self.output_dim = int(feature_dim)
        self.k = int(k)
        self.model = DGCNGn(
            emb_size=self.output_dim,
            primitives=False,
            embedding=True,
            mode=0,
            num_channels=3,
            k=self.k,
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.shape[1] < self.k:
            raise ValueError(
                f"DGCNN requires at least dgcnn_k={self.k} points, got {xyz.shape[1]}"
            )
        features, _ = self.model(xyz.transpose(1, 2).contiguous())
        return features.transpose(1, 2).contiguous()


class _PointTransformerFeatureAdapter(nn.Module):
    def __init__(self, k: int, width: int, depth: int):
        super().__init__()
        self.model = PointTransformerSegmenter(
            num_classes=1,
            use_constraints=False,
            k=k,
            width=width,
            depth=depth,
        )
        self.model.head = _without_final_layer(self.model.head)
        self.output_dim = 2 * int(width)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.model(xyz)


class _PointMambaFeatureAdapter(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        group_size: int,
        width: int,
        depth: int,
    ):
        super().__init__()
        self.model = PointMambaSegmenter(
            num_classes=1,
            use_constraints=False,
            num_tokens=num_tokens,
            group_size=group_size,
            width=width,
            depth=depth,
        )
        self.model.head = _without_final_layer(self.model.head)
        self.output_dim = 2 * int(width)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.model(xyz)


class _PointNeXtFeatureAdapter(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.model = PointNeXtSegmenter(
            num_classes=1,
            use_constraints=False,
            k=k,
        )
        self.model.head = _without_final_layer(self.model.head)
        self.output_dim = 64

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.model(xyz)


class _PointMLPFeatureAdapter(nn.Module):
    def __init__(self, group_size: int):
        super().__init__()
        self.model = PointMLPSegmenter(
            num_classes=1,
            use_constraints=False,
            group_size=group_size,
        )
        self.model.head = _without_final_layer(self.model.head)
        self.output_dim = 64

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.model(xyz)


def _build_feature_backbone(config: Mapping[str, Any]) -> nn.Module:
    model_name = str(config["model"])
    feature_dim = int(config["feature_dim"])
    if model_name == "pointnet":
        return _PointNetFeatureAdapter(feature_dim)
    if model_name == "pointnet2":
        return _PointNet2FeatureAdapter(feature_dim)
    if model_name == "attn3dgcn":
        return _Attn3DGCNFeatureAdapter(
            feature_dim,
            config["attn_neighbors"],
            config["attn_k"],
        )
    if model_name == "dgcnn":
        return _DGCNNFeatureAdapter(feature_dim, config["dgcnn_k"])
    if model_name == "pointtransformer":
        return _PointTransformerFeatureAdapter(
            config["pointtransformer_k"],
            config["pointtransformer_width"],
            config["pointtransformer_depth"],
        )
    if model_name == "pointmamba":
        return _PointMambaFeatureAdapter(
            config["pointmamba_tokens"],
            config["pointmamba_group_size"],
            config["pointmamba_width"],
            config["pointmamba_depth"],
        )
    if model_name == "pointnext":
        return _PointNeXtFeatureAdapter(config["pointnext_k"])
    if model_name == "pointmlp":
        return _PointMLPFeatureAdapter(config["pointmlp_group_size"])
    raise AssertionError(f"unhandled direct baseline model: {model_name}")


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _PerPointProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if input_dim == output_dim:
            self.layers = nn.Identity()
        else:
            self.layers = nn.Sequential(
                nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False),
                nn.GroupNorm(_group_count(output_dim), output_dim),
                nn.GELU(),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel_first = features.transpose(1, 2).contiguous()
        projected = self.layers(channel_first)
        return projected.transpose(1, 2).contiguous()


class _DirectHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.layers(features.transpose(1, 2).contiguous())
        return output.transpose(1, 2).contiguous()


class Stage1DirectBaseline(nn.Module):
    """Shared XYZ backbone with four independent per-point constraint heads."""

    def __init__(self, config: Mapping[str, Any] | Any):
        super().__init__()
        self.model_config = stage1_direct_model_config(config)
        self.backbone = _build_feature_backbone(self.model_config)
        native_dim = int(self.backbone.output_dim)
        feature_dim = int(self.model_config["feature_dim"])
        hidden_dim = int(self.model_config["head_hidden_dim"])
        dropout = float(self.model_config["head_dropout"])
        self.feature_projection = _PerPointProjection(native_dim, feature_dim)
        self.primitive_head = _DirectHead(
            feature_dim, hidden_dim, N_PRIMITIVES, dropout
        )
        self.direction_head = _DirectHead(feature_dim, hidden_dim, 3, dropout)
        self.dimension_head = _DirectHead(feature_dim, hidden_dim, 1, dropout)
        self.location_head = _DirectHead(feature_dim, hidden_dim, 3, dropout)

    def forward(self, xyz: torch.Tensor) -> dict[str, torch.Tensor]:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"expected xyz [B, N, 3], got {tuple(xyz.shape)}")
        if xyz.shape[1] == 0:
            raise ValueError("point cloud must contain at least one point")
        features = self.feature_projection(self.backbone(xyz))
        primitive_logits = self.primitive_head(features)
        direction = F.normalize(
            self.direction_head(features), dim=-1, eps=1e-6
        )
        dimension = F.softplus(self.dimension_head(features)).squeeze(-1)
        location = self.location_head(features)
        return {
            "pmt_logits": primitive_logits,
            "log_pmt": F.log_softmax(primitive_logits, dim=-1),
            "mad": direction,
            "dim": dimension,
            "loc": location,
        }

    @torch.no_grad()
    def predict_constraints(
        self, xyz: torch.Tensor, *, as_tensor: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        predictions = self.forward(xyz)
        constraints = finalize_direct_constraints(predictions)
        return constraints_to_tensor(constraints) if as_tensor else constraints


def finalize_direct_constraints(
    predictions: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply only representation rules; no clustering or geometric fitting."""
    required = {"log_pmt", "mad", "dim", "loc"}
    missing = sorted(required.difference(predictions))
    if missing:
        raise ValueError(f"direct prediction is missing fields: {missing}")

    log_pmt = predictions["log_pmt"]
    direction_raw = predictions["mad"]
    dimension_raw = predictions["dim"]
    location_raw = predictions["loc"]
    if log_pmt.ndim != 3 or log_pmt.shape[-1] != N_PRIMITIVES:
        raise ValueError(f"expected log_pmt [B, N, 5], got {tuple(log_pmt.shape)}")
    expected_bn = tuple(log_pmt.shape[:2])
    if tuple(direction_raw.shape) != (*expected_bn, 3):
        raise ValueError("direction prediction shape does not match primitive prediction")
    if tuple(dimension_raw.shape) != expected_bn:
        raise ValueError("dimension prediction shape does not match primitive prediction")
    if tuple(location_raw.shape) != (*expected_bn, 3):
        raise ValueError("location prediction shape does not match primitive prediction")

    primitive_index = log_pmt.argmax(dim=-1)
    primitive_type = F.one_hot(
        primitive_index, num_classes=N_PRIMITIVES
    ).to(dtype=direction_raw.dtype)
    direction = canonicalize_directions(direction_raw)
    invalid_direction = direction.new_tensor(INVALID_DIRECTION).view(1, 1, 3)
    direction_valid = (primitive_index == 0) | (primitive_index == 1) | (
        primitive_index == 2
    )
    direction = torch.where(
        direction_valid.unsqueeze(-1), direction, invalid_direction
    )

    dimension_valid = (primitive_index == 1) | (primitive_index == 2) | (
        primitive_index == 3
    )
    dimension = torch.where(
        dimension_valid, dimension_raw, dimension_raw.new_full((), -1.0)
    )
    location = torch.where(
        (primitive_index != 4).unsqueeze(-1), location_raw, torch.zeros_like(location_raw)
    )
    return {
        "primitive_type": primitive_type,
        "direction": direction,
        "dimension": dimension,
        "location": location,
    }


def build_stage1_direct_baseline(
    config: Mapping[str, Any] | Any,
) -> Stage1DirectBaseline:
    return Stage1DirectBaseline(config)


__all__ = [
    "DIRECT_BASELINE_MODEL_NAMES",
    "Stage1DirectBaseline",
    "build_stage1_direct_baseline",
    "finalize_direct_constraints",
    "stage1_direct_model_config",
]
