from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from functional.constraints import (
    CLUSTER_METHODS,
    assemble_constraints_from_stage1,
    constraints_to_tensor,
)
from functional.point_features import build_stage1_input_features, stage1_feature_dim
from networks.cst_pred_wrapper import CstPredWrapper


class FrozenStage1ConstraintExtractor(nn.Module):
    """
    Frozen Stage 1 wrapper used by Stage 2.

    It loads the primitive/cluster predictor, disables gradients, and converts
    model outputs into the per-point constraint representation required by
    classification and segmentation models.
    """

    def __init__(
        self,
        model_name: str = "pointnet2",
        checkpoint: Optional[str] = None,
        cluster_bandwidth: float = 0.35,
        cluster_method: str = "meanshift",
        mean_shift_quantile: float = 0.015,
        mean_shift_iterations: int = 50,
        mean_shift_max_clusters: int = 128,
        mean_shift_bandwidth: Optional[float] = None,
        normal_k: int = 16,
        channel_mid: int = 128,
        cluster_dim: int = 32,
        use_extra_features: bool = False,
        feature_k: int = 16,
        use_pca_normals_for_fitting: bool = True,
        use_prediction_initialization: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.cluster_bandwidth = cluster_bandwidth
        if cluster_method not in CLUSTER_METHODS:
            raise ValueError(
                f"unsupported cluster_method={cluster_method!r}; expected one of {CLUSTER_METHODS}"
            )
        self.cluster_method = cluster_method
        self.mean_shift_quantile = float(mean_shift_quantile)
        self.mean_shift_iterations = int(mean_shift_iterations)
        self.mean_shift_max_clusters = int(mean_shift_max_clusters)
        self.mean_shift_bandwidth = mean_shift_bandwidth
        self.normal_k = normal_k
        self.use_extra_features = bool(use_extra_features)
        self.feature_k = int(feature_k)
        self.use_pca_normals_for_fitting = bool(use_pca_normals_for_fitting)
        self.use_prediction_initialization = bool(use_prediction_initialization)
        self.model = CstPredWrapper(
            embedding_model_name=model_name,
            channel_fea=stage1_feature_dim(self.use_extra_features),
            channel_mid=channel_mid,
            channel_out=cluster_dim,
            n_prim_type=5,
        )
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        self.freeze()

    def load_checkpoint(self, checkpoint: str) -> None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        checkpoint_args = (
            checkpoint_data.get("args", {})
            if isinstance(checkpoint_data, dict)
            else {}
        )
        if not isinstance(checkpoint_args, dict):
            checkpoint_args = {}
        state = checkpoint_data
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"invalid Stage 1 model state: {checkpoint}")

        current = self.model.state_dict()
        missing = sorted(key for key in current if key not in state)
        unexpected = sorted(key for key in state if key not in current)
        shape_mismatch = {}
        for key in set(state).intersection(current):
            incoming = state[key]
            if not torch.is_tensor(incoming):
                shape_mismatch[key] = ("not-a-tensor", tuple(current[key].shape))
            elif tuple(incoming.shape) != tuple(current[key].shape):
                shape_mismatch[key] = (
                    tuple(incoming.shape), tuple(current[key].shape)
                )
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and key not in shape_mismatch
        }
        print(f"Stage 1 missing_keys: {missing}")
        print(f"Stage 1 unexpected_keys: {unexpected}")
        print(f"Stage 1 shape_mismatch: {shape_mismatch}")
        print(f"Stage 1 common weights loaded completely: {not missing and not shape_mismatch}")
        self.model.load_state_dict(compatible, strict=False)

        geometry_prefixes = (
            "geometry_decoder.", "mad_head.", "dim_head.", "loc_head."
        )
        geometry_incomplete = any(
            key.startswith(geometry_prefixes) for key in (*missing, *shape_mismatch)
        )
        checkpoint_phase = str(checkpoint_args.get("train_phase", ""))
        if self.use_prediction_initialization and (
            checkpoint_phase not in {"geometry", "joint"} or geometry_incomplete
        ):
            self.use_prediction_initialization = False
            print(
                "WARNING: fitting initialization disabled because the checkpoint "
                "does not contain a complete trained geometry/joint stage"
            )

    def freeze(self) -> None:
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def predict_raw(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        features = None
        if self.use_extra_features:
            features = build_stage1_input_features(
                xyz,
                use_curvature=True,
                use_density=True,
                k=self.feature_k,
            )
        return self.model(xyz, features)

    @torch.no_grad()
    def forward(
        self,
        xyz: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        prediction = self.predict_raw(xyz)
        constraints = assemble_constraints_from_stage1(
            xyz=xyz,
            cluster_embedding=prediction["embedding"],
            log_primitive=prediction["log_pmt"],
            cluster_bandwidth=self.cluster_bandwidth,
            cluster_method=self.cluster_method,
            mean_shift_quantile=self.mean_shift_quantile,
            mean_shift_iterations=self.mean_shift_iterations,
            mean_shift_max_clusters=self.mean_shift_max_clusters,
            mean_shift_bandwidth=self.mean_shift_bandwidth,
            normal_k=self.normal_k,
            use_pca_normals_for_fitting=self.use_pca_normals_for_fitting,
            mad_prediction=prediction["mad"],
            dim_prediction=prediction["dim"],
            loc_prediction=prediction["loc"],
            use_prediction_initialization=self.use_prediction_initialization,
        )
        if return_dict:
            return constraints
        return constraints_to_tensor(constraints)
