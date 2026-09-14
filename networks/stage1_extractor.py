from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from functional.constraints import (
    constraints_to_tensor,
)
from functional.direct_constraints import direct_constraints, validate_direct_checkpoint
from functional.point_features import stage1_forward, stage1_feature_dim
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
        channel_mid: int = 128,
        cluster_dim: int = 32,
        use_extra_features: bool = False,
        feature_k: int = 16,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_extra_features = bool(use_extra_features)
        self.feature_k = int(feature_k)
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

        validate_direct_checkpoint(checkpoint_args)
        self.model.load_state_dict(state, strict=True)

    def freeze(self) -> None:
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def predict_raw(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        return stage1_forward(self.model, xyz, use_extra_features=self.use_extra_features, feature_k=self.feature_k)

    @torch.no_grad()
    def forward(
        self,
        xyz: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        prediction = self.predict_raw(xyz)
        constraints = direct_constraints(prediction)
        if return_dict:
            return constraints
        return constraints_to_tensor(constraints)
