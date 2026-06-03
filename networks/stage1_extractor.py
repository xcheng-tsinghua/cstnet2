from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from functional.constraints import assemble_constraints_from_stage1, constraints_to_tensor
from networks.cst_pred_wrapper import CstPredWrapper


class FrozenStage1ConstraintExtractor(nn.Module):
    """
    Frozen Stage 1 wrapper used by Stage 2.

    It loads the primitive/cluster predictor, disables gradients, and converts
    model outputs into the per-point constraint representation required by
    classification and diffusion models.
    """

    def __init__(
        self,
        model_name: str = "pointnet2",
        checkpoint: Optional[str] = None,
        cluster_bandwidth: float = 0.35,
        normal_k: int = 16,
        channel_mid: int = 128,
        cluster_dim: int = 32,
    ):
        super().__init__()
        self.model_name = model_name
        self.cluster_bandwidth = cluster_bandwidth
        self.normal_k = normal_k
        self.model = CstPredWrapper(
            embedding_model_name=model_name,
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
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)

    def freeze(self) -> None:
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def predict_raw(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        return self.model(xyz)

    @torch.no_grad()
    def forward(
        self,
        xyz: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        cluster_embedding, log_primitive = self.predict_raw(xyz)
        constraints = assemble_constraints_from_stage1(
            xyz=xyz,
            cluster_embedding=cluster_embedding,
            log_primitive=log_primitive,
            normals=normals,
            cluster_bandwidth=self.cluster_bandwidth,
            normal_k=self.normal_k,
        )
        if return_dict:
            return constraints
        return constraints_to_tensor(constraints)
