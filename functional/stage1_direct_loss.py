"""Pure four-component supervision for XYZ-only Stage 1 direct baselines."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from functional.finite_checks import assert_finite_tensors


def _primitive_mask(
    primitive: torch.Tensor, valid_types: tuple[int, ...]
) -> torch.Tensor:
    mask = torch.zeros_like(primitive, dtype=torch.bool)
    for primitive_type in valid_types:
        mask |= primitive == primitive_type
    return mask


def _masked_direction_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Mask before arithmetic so invalid targets cannot contaminate gradients.
    prediction = F.normalize(torch.where(mask[..., None], prediction, 0.0), dim=-1, eps=1e-6)
    target = F.normalize(torch.where(mask[..., None], target, 0.0), dim=-1, eps=1e-6)
    direct = (prediction - target).pow(2).mean(dim=-1)
    flipped = (prediction + target).pow(2).mean(dim=-1)
    errors = torch.where(mask, torch.minimum(direct, flipped), 0.0)
    if errors.dtype in (torch.float16, torch.bfloat16):
        errors = errors.float()
    return errors.sum() / mask.sum().clamp_min(1)


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    expanded_mask = mask if prediction.ndim == mask.ndim else mask[..., None]
    prediction = torch.where(expanded_mask, prediction, 0.0)
    target = torch.where(expanded_mask, target, 0.0)
    if prediction.dtype in (torch.float16, torch.bfloat16):
        prediction = prediction.float()
    if target.dtype in (torch.float16, torch.bfloat16):
        target = target.float()
    errors = (prediction - target).square()
    if errors.ndim > mask.ndim:
        errors = errors.mean(dim=-1)
    return errors.sum() / mask.sum().clamp_min(1)


def direct_constraint_loss(
    predictions: Mapping[str, torch.Tensor],
    pmt_gt: torch.Tensor,
    mad_gt: torch.Tensor,
    dim_gt: torch.Tensor,
    loc_gt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute only direct component losses, with no clustering regularizers."""
    required = {"log_pmt", "mad", "dim", "loc"}
    missing = sorted(required.difference(predictions))
    if missing:
        raise ValueError(f"direct prediction is missing fields: {missing}")

    log_pmt = predictions["log_pmt"]
    mad_pred = predictions["mad"]
    dim_pred = predictions["dim"]
    loc_pred = predictions["loc"]
    expected_bn = tuple(pmt_gt.shape)
    expected_shapes = {
        "log_pmt": (*expected_bn, 5),
        "mad": (*expected_bn, 3),
        "dim": expected_bn,
        "loc": (*expected_bn, 3),
    }
    actual = {
        "log_pmt": tuple(log_pmt.shape),
        "mad": tuple(mad_pred.shape),
        "dim": tuple(dim_pred.shape),
        "loc": tuple(loc_pred.shape),
    }
    mismatched = {
        name: (actual[name], expected)
        for name, expected in expected_shapes.items()
        if actual[name] != expected
    }
    if mismatched:
        raise ValueError(f"direct prediction/target shape mismatch: {mismatched}")

    pmt_loss = F.nll_loss(log_pmt.reshape(-1, 5), pmt_gt.reshape(-1).long())
    mad_mask = _primitive_mask(pmt_gt, (0, 1, 2))
    dim_mask = _primitive_mask(pmt_gt, (1, 2, 3))
    loc_mask = _primitive_mask(pmt_gt, (0, 1, 2, 3))
    mad_loss = _masked_direction_mse(mad_pred, mad_gt, mad_mask)
    dim_loss = _masked_mse(dim_pred, dim_gt, dim_mask)
    loc_loss = _masked_mse(loc_pred, loc_gt, loc_mask)

    total = pmt_loss + mad_loss + dim_loss + loc_loss
    loss_dict: dict[str, torch.Tensor] = {
        "loss_all": total,
        "pmt_loss": pmt_loss,
        "mad_loss": mad_loss,
        "dim_loss": dim_loss,
        "loc_loss": loc_loss,
    }
    assert_finite_tensors(loss_dict, "Stage 1 direct losses")
    return total, loss_dict


__all__ = [
    "direct_constraint_loss",
]
