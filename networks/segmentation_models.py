from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from functional.constraints import CONSTRAINT_DIM
from networks.attn_3dgcn import Attn3DGcnPointEmbedding
from networks.dgcnn_gn import DGCNGn
from networks.point_net import PointNetSeg
from networks.point_net2 import PointNet2Seg
from networks.point_mamba import PointMambaSegmenter
from networks.point_transformer import PointTransformerSegmenter
from networks.pointmlp import PointMLPSegmenter
from networks.pointnext import PointNeXtSegmenter
from networks.stage2_segmentation import Stage2SegmentationModel


DEFAULT_SEGMENTATION_MODEL = "constraint_aware"
SEGMENTATION_MODEL_NAMES = (
    DEFAULT_SEGMENTATION_MODEL,
    "pointnet",
    "pointnet2",
    "dgcnn",
    "attn3dgcn",
    "pointtransformer",
    "pointmamba",
    "pointnext",
    "pointmlp",
)
BASELINE_SEGMENTATION_MODELS = SEGMENTATION_MODEL_NAMES[1:]


def _config_value(source: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def segmentation_model_config(source: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return checkpoint-relevant model selection and constraint settings.

    Baseline internals deliberately come from defaults in their model files.
    """
    model_name = str(
        _config_value(source, "model", DEFAULT_SEGMENTATION_MODEL)
        or DEFAULT_SEGMENTATION_MODEL
    ).lower()
    if model_name not in SEGMENTATION_MODEL_NAMES:
        raise ValueError(
            f"unknown segmentation model {model_name!r}; "
            f"expected one of {SEGMENTATION_MODEL_NAMES}"
        )

    config: dict[str, Any] = {"model": model_name}
    if model_name == "constraint_aware":
        if bool(_config_value(source, "baseline_use_constraints", False)):
            raise ValueError(
                "baseline_use_constraints is only valid for baseline segmentation models"
            )
        config.update(
            feature_dim=int(_config_value(source, "feature_dim", 64)),
            norm_type=str(_config_value(source, "norm_type", "ln")),
        )
    else:
        config["baseline_use_constraints"] = bool(
            _config_value(source, "baseline_use_constraints", False)
        )
    return config


def _validate_xyz(xyz: torch.Tensor) -> None:
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"expected xyz [B, N, 3], got {tuple(xyz.shape)}")


def _baseline_inputs(
    xyz: torch.Tensor,
    constraints: torch.Tensor | None,
    use_constraints: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return channel-first XYZ(+constraints) and optional constraint features."""
    _validate_xyz(xyz)
    constraint_features = None
    if use_constraints:
        expected_shape = (xyz.shape[0], xyz.shape[1], CONSTRAINT_DIM)
        if constraints is None or tuple(constraints.shape) != expected_shape:
            actual_shape = None if constraints is None else tuple(constraints.shape)
            raise ValueError(
                f"constraint-enabled baseline expects constraints {expected_shape}, "
                f"got {actual_shape}"
            )
        constraint_features = constraints
        points = torch.cat([xyz, constraints], dim=-1)
    else:
        points = xyz
    channel_first = points.transpose(1, 2).contiguous()
    constraint_channel_first = (
        None
        if constraint_features is None
        else constraint_features.transpose(1, 2).contiguous()
    )
    return channel_first, constraint_channel_first


class PointNetSegmentationAdapter(nn.Module):
    """Adapter for the existing ShapeNet-style PointNet segmentation head."""

    def __init__(self, num_classes: int, use_constraints: bool = False):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        input_channels = 3 + (CONSTRAINT_DIM if self.use_constraints else 0)
        self.model = PointNetSeg(
            part_num=num_classes,
            normal_channel=False,
            input_channels=input_channels,
        )

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor | None = None,
        constraint_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del constraint_masks
        point_input, _ = _baseline_inputs(xyz, constraints, self.use_constraints)
        # The original implementation reserves 16 channels for a ShapeNet
        # object-category one-hot vector. MFCAD++ has no such sample-level
        # category input, so the neutral all-zero vector is used consistently.
        object_category = xyz.new_zeros((xyz.shape[0], 1, 16))
        logits, _ = self.model(point_input, object_category)
        return logits


class PointNet2SegmentationAdapter(nn.Module):
    """Adapter for the existing PointNet++ segmenter."""

    def __init__(self, num_classes: int, use_constraints: bool = False):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        input_channels = 3 + (CONSTRAINT_DIM if self.use_constraints else 0)
        self.model = PointNet2Seg(
            num_classes=num_classes,
            normal_channel=False,
            input_channels=input_channels,
        )

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor | None = None,
        constraint_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del constraint_masks
        point_input, _ = _baseline_inputs(xyz, constraints, self.use_constraints)
        logits = self.model(point_input)
        return logits.transpose(1, 2).contiguous()


class DGCNNSegmentationAdapter(nn.Module):
    """Adapter using the existing DGCNN-GN embedding head."""

    def __init__(self, num_classes: int, k: int = 20, use_constraints: bool = False):
        super().__init__()
        if k <= 0:
            raise ValueError("dgcnn_k must be positive")
        self.k = int(k)
        self.use_constraints = bool(use_constraints)
        self.model = DGCNGn(
            emb_size=num_classes,
            primitives=False,
            embedding=True,
            mode=0,
            num_channels=3 + (CONSTRAINT_DIM if self.use_constraints else 0),
            k=self.k,
        )

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor | None = None,
        constraint_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del constraint_masks
        point_input, _ = _baseline_inputs(xyz, constraints, self.use_constraints)
        if xyz.shape[1] < self.k:
            raise ValueError(
                f"DGCNN requires at least dgcnn_k={self.k} points, got {xyz.shape[1]}"
            )
        logits, _ = self.model(point_input)
        return logits.transpose(1, 2).contiguous()


class Attn3DGCNSegmentationAdapter(nn.Module):
    """Adapter around the existing Attention 3DGCN encoder."""

    def __init__(
        self,
        num_classes: int,
        n_neighbors: int = 20,
        attn_k: int = 16,
        feature_dim: int = 128,
        use_constraints: bool = False,
    ):
        super().__init__()
        if n_neighbors <= 0 or attn_k <= 0:
            raise ValueError("attn_neighbors and attn_k must be positive")
        self.n_neighbors = int(n_neighbors)
        self.attn_k = int(attn_k)
        self.use_constraints = bool(use_constraints)
        self.encoder = Attn3DGcnPointEmbedding(
            channel_coord=3,
            channel_fea=CONSTRAINT_DIM if self.use_constraints else 0,
            channel_out=feature_dim,
            n_neighbor=self.n_neighbors,
            attn_k=self.attn_k,
        )
        self.head = nn.Conv1d(feature_dim, num_classes, kernel_size=1)

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor | None = None,
        constraint_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del constraint_masks
        _, constraint_features = _baseline_inputs(
            xyz, constraints, self.use_constraints
        )
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
        features = self.encoder(
            xyz.transpose(1, 2).contiguous(),
            constraint_features,
        )
        return self.head(features).transpose(1, 2).contiguous()


def build_segmentation_model(
    num_classes: int,
    config: Mapping[str, Any] | Any,
) -> nn.Module:
    """Build a registered segmentation model with the common Stage 2 API."""
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    model_config = segmentation_model_config(config)
    model_name = model_config["model"]

    if model_name == "constraint_aware":
        return Stage2SegmentationModel(
            num_classes=num_classes,
            feature_dim=model_config["feature_dim"],
            norm_type=model_config["norm_type"],
        )
    if model_name == "pointnet":
        return PointNetSegmentationAdapter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
        )
    if model_name == "pointnet2":
        return PointNet2SegmentationAdapter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
        )
    if model_name == "dgcnn":
        return DGCNNSegmentationAdapter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
        )
    if model_name == "attn3dgcn":
        return Attn3DGCNSegmentationAdapter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
        )
    if model_name == "pointtransformer":
        return PointTransformerSegmenter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
            constraint_dim=CONSTRAINT_DIM,
        )
    if model_name == "pointmamba":
        return PointMambaSegmenter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
            constraint_dim=CONSTRAINT_DIM,
        )
    if model_name == "pointnext":
        return PointNeXtSegmenter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
            constraint_dim=CONSTRAINT_DIM,
        )
    if model_name == "pointmlp":
        return PointMLPSegmenter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
            constraint_dim=CONSTRAINT_DIM,
        )
    raise AssertionError(f"unhandled segmentation model: {model_name}")
