from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from functional.constraints import CONSTRAINT_DIM
from networks.attn_3dgcn import Attn3DGcnPointEmbedding
from networks.dgcnn_gn import DGCNNEncoderGn
from networks.point_net import PointNet
from networks.point_net2 import PointNet2Cls
from networks.stage2 import (
    CstNetStage2Classifier,
    CstNetStage2ClassifierDiscriminative,
    CstNetStage2ClassifierTokenFusion,
)


DEFAULT_CLASSIFICATION_MODEL = "constraint_aware"
CLASSIFICATION_MODEL_NAMES = (
    DEFAULT_CLASSIFICATION_MODEL,
    "pointnet",
    "pointnet2",
    "dgcnn",
    "attn3dgcn",
)
BASELINE_CLASSIFICATION_MODELS = CLASSIFICATION_MODEL_NAMES[1:]
CONSTRAINT_AWARE_VARIANTS = ("baseline", "discriminative", "token_fusion")


def _config_value(source: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def classification_model_config(
    source: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Return only parameters that define the classification architecture."""
    model_name = str(
        _config_value(source, "model", DEFAULT_CLASSIFICATION_MODEL)
        or DEFAULT_CLASSIFICATION_MODEL
    ).lower()
    if model_name not in CLASSIFICATION_MODEL_NAMES:
        raise ValueError(
            f"unknown classification model {model_name!r}; "
            f"expected one of {CLASSIFICATION_MODEL_NAMES}"
        )

    config: dict[str, Any] = {"model": model_name}
    if model_name == DEFAULT_CLASSIFICATION_MODEL:
        if _as_bool(_config_value(source, "baseline_use_constraints", False)):
            raise ValueError(
                "baseline_use_constraints is only valid for baseline "
                "classification models"
            )
        variant = str(_config_value(source, "stage2_variant", "baseline"))
        if variant not in CONSTRAINT_AWARE_VARIANTS:
            raise ValueError(
                f"unknown constraint-aware classification variant {variant!r}; "
                f"expected one of {CONSTRAINT_AWARE_VARIANTS}"
            )
        config["stage2_variant"] = variant
        if variant != "baseline":
            config["stage2_norm"] = str(
                _config_value(source, "stage2_norm", "ln")
            )
        if variant == "token_fusion":
            config.update(
                token_dim=int(_config_value(source, "token_dim", 256)),
                transformer_layers=int(
                    _config_value(source, "transformer_layers", 3)
                ),
                transformer_heads=int(
                    _config_value(source, "transformer_heads", 8)
                ),
                token_dropout=float(_config_value(source, "token_dropout", 0.1)),
                stream_dropout=float(
                    _config_value(source, "stream_dropout", 0.1)
                ),
                use_stats_token=_as_bool(
                    _config_value(source, "use_stats_token", False)
                ),
            )
    else:
        if str(_config_value(source, "stage2_variant", "baseline")) != "baseline":
            raise ValueError(
                "stage2_variant is only valid for the constraint_aware model"
            )
        config["baseline_use_constraints"] = _as_bool(
            _config_value(source, "baseline_use_constraints", False)
        )
        if model_name == "dgcnn":
            config["dgcnn_k"] = int(_config_value(source, "dgcnn_k", 20))
        elif model_name == "attn3dgcn":
            config.update(
                attn_neighbors=int(_config_value(source, "attn_neighbors", 20)),
                attn_k=int(_config_value(source, "attn_k", 16)),
            )
    return config


def classification_model_uses_constraints(config: Mapping[str, Any]) -> bool:
    return config["model"] == DEFAULT_CLASSIFICATION_MODEL or bool(
        config.get("baseline_use_constraints", False)
    )


def classification_run_name(config: Mapping[str, Any]) -> str:
    model_name = str(config["model"])
    if model_name == DEFAULT_CLASSIFICATION_MODEL:
        variant = str(config.get("stage2_variant", "baseline"))
        return model_name if variant == "baseline" else f"{model_name}_{variant}"
    if bool(config.get("baseline_use_constraints", False)):
        return f"{model_name}_constraints"
    return model_name


def _validate_inputs(
    xyz: torch.Tensor,
    constraints: torch.Tensor | None,
    use_constraints: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"expected xyz [B, N, 3], got {tuple(xyz.shape)}")
    constraint_features = None
    if use_constraints:
        expected = (xyz.shape[0], xyz.shape[1], CONSTRAINT_DIM)
        if constraints is None or tuple(constraints.shape) != expected:
            actual = None if constraints is None else tuple(constraints.shape)
            raise ValueError(
                f"constraint-enabled baseline expects constraints {expected}, "
                f"got {actual}"
            )
        constraint_features = constraints.transpose(1, 2).contiguous()
    return xyz.transpose(1, 2).contiguous(), constraint_features


class PointNetClassificationAdapter(nn.Module):
    def __init__(self, num_classes: int, use_constraints: bool = False):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        self.model = PointNet(
            n_classes=num_classes,
            fea_channel=CONSTRAINT_DIM if self.use_constraints else 0,
        )

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xyz_cf, constraint_cf = _validate_inputs(
            xyz, constraints, self.use_constraints
        )
        return self.model(xyz_cf, constraint_cf)


class PointNet2ClassificationAdapter(nn.Module):
    def __init__(self, num_classes: int, use_constraints: bool = False):
        super().__init__()
        self.use_constraints = bool(use_constraints)
        input_channels = 3 + (CONSTRAINT_DIM if self.use_constraints else 0)
        self.model = PointNet2Cls(
            num_class=num_classes,
            input_channels=input_channels,
        )

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xyz_cf, constraint_cf = _validate_inputs(
            xyz, constraints, self.use_constraints
        )
        point_features = (
            xyz_cf
            if constraint_cf is None
            else torch.cat([xyz_cf, constraint_cf], dim=1)
        )
        return self.model(point_features)


class DGCNNClassificationAdapter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        k: int = 20,
        use_constraints: bool = False,
    ):
        super().__init__()
        if k <= 0:
            raise ValueError("dgcnn_k must be positive")
        self.k = int(k)
        self.use_constraints = bool(use_constraints)
        input_channels = 3 + (CONSTRAINT_DIM if self.use_constraints else 0)
        self.encoder = DGCNNEncoderGn(
            mode=0,
            input_channels=input_channels,
            k=self.k,
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xyz_cf, constraint_cf = _validate_inputs(
            xyz, constraints, self.use_constraints
        )
        if xyz.shape[1] < self.k:
            raise ValueError(
                f"DGCNN requires at least dgcnn_k={self.k} points, "
                f"got {xyz.shape[1]}"
            )
        point_features = (
            xyz_cf
            if constraint_cf is None
            else torch.cat([xyz_cf, constraint_cf], dim=1)
        )
        global_features, _ = self.encoder(point_features)
        return F.log_softmax(self.head(global_features), dim=-1)


class Attn3DGCNClassificationAdapter(nn.Module):
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
        self.head = nn.Sequential(
            nn.Linear(2 * feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xyz_cf, constraint_cf = _validate_inputs(
            xyz, constraints, self.use_constraints
        )
        if xyz.shape[1] <= self.n_neighbors:
            raise ValueError(
                "Attention 3DGCN requires more points than attn_neighbors; "
                f"got N={xyz.shape[1]}, attn_neighbors={self.n_neighbors}"
            )
        if xyz.shape[1] < self.attn_k:
            raise ValueError(
                f"Attention 3DGCN requires at least attn_k={self.attn_k} "
                f"points, got {xyz.shape[1]}"
            )
        features = self.encoder(xyz_cf, constraint_cf)
        pooled = torch.cat(
            [features.amax(dim=2), features.mean(dim=2)], dim=1
        )
        return F.log_softmax(self.head(pooled), dim=-1)


def build_classification_model(
    num_classes: int,
    config: Mapping[str, Any] | Any,
) -> nn.Module:
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    model_config = classification_model_config(config)
    model_name = model_config["model"]

    if model_name == DEFAULT_CLASSIFICATION_MODEL:
        variant = model_config["stage2_variant"]
        if variant == "baseline":
            return CstNetStage2Classifier(n_classes=num_classes)
        if variant == "discriminative":
            return CstNetStage2ClassifierDiscriminative(
                n_classes=num_classes,
                norm_type=model_config["stage2_norm"],
            )
        if variant == "token_fusion":
            return CstNetStage2ClassifierTokenFusion(
                n_classes=num_classes,
                norm_type=model_config["stage2_norm"],
                token_dim=model_config["token_dim"],
                transformer_layers=model_config["transformer_layers"],
                transformer_heads=model_config["transformer_heads"],
                dropout=model_config["token_dropout"],
                stream_dropout=model_config["stream_dropout"],
                use_stats_token=model_config["use_stats_token"],
            )
        raise AssertionError(f"unhandled constraint-aware variant: {variant}")
    if model_name == "pointnet":
        return PointNetClassificationAdapter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
        )
    if model_name == "pointnet2":
        return PointNet2ClassificationAdapter(
            num_classes,
            use_constraints=model_config["baseline_use_constraints"],
        )
    if model_name == "dgcnn":
        return DGCNNClassificationAdapter(
            num_classes,
            k=model_config["dgcnn_k"],
            use_constraints=model_config["baseline_use_constraints"],
        )
    if model_name == "attn3dgcn":
        return Attn3DGCNClassificationAdapter(
            num_classes,
            n_neighbors=model_config["attn_neighbors"],
            attn_k=model_config["attn_k"],
            use_constraints=model_config["baseline_use_constraints"],
        )
    raise AssertionError(f"unhandled classification model: {model_name}")
