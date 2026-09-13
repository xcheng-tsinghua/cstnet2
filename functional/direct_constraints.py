"""Canonical per-point constraints from the Stage 1 prediction heads."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from functional.constraints import canonicalize_directions, zero_invalid_constraint_components


CONSTRAINT_ROUTE = "direct_mlp_v1"


def validate_direct_checkpoint(args, *, require_geometry=True):
    if args.get("constraint_route") != CONSTRAINT_ROUTE:
        raise ValueError("This model requires a direct_mlp_v1 checkpoint; retrain with train_cst_pred.py")
    if require_geometry and args.get("train_phase") not in {"geometry", "joint"}:
        raise ValueError("Final constraints require a trained geometry or joint checkpoint")


def direct_constraints(prediction):
    """Return the four components; no clustering, fitting, or XYZ is needed."""
    primitive = prediction["log_pmt"].argmax(dim=-1)
    raw_direction = prediction["mad"]
    fallback = torch.zeros_like(raw_direction)
    fallback[..., 2] = 1.0
    raw_direction = torch.where(
        raw_direction.norm(dim=-1, keepdim=True) > 1e-7, raw_direction, fallback
    )
    direction = canonicalize_directions(raw_direction, eps=1e-7)
    dimension = prediction["dim"].clamp_min(1e-6)
    dimension = torch.where(
        primitive == 2, dimension.clamp(max=math.pi / 2 - 1e-4), dimension
    )
    location = prediction["loc"]
    axial = (location * direction).sum(dim=-1, keepdim=True) * direction
    location = torch.where((primitive == 0).unsqueeze(-1), axial, location)
    location = torch.where((primitive == 1).unsqueeze(-1), location - axial, location)
    location = torch.where((primitive == 4).unsqueeze(-1), torch.zeros_like(location), location)
    direction, dimension = zero_invalid_constraint_components(primitive, direction, dimension)
    return {
        "primitive_type": F.one_hot(primitive, num_classes=5).to(location.dtype),
        "direction": direction,
        "dimension": dimension,
        "location": location,
    }
