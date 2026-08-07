"""Pure four-component supervision for XYZ-only Stage 1 direct baselines."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from functional.constraints import canonicalize_directions


DIRECT_LOSS_NAMES = ("pmt", "mad", "dim", "loc")
DEFAULT_DIRECT_LOSS_WEIGHTS = {
    "w_pmt": 1.0,
    "w_mad": 0.02,
    "w_dim": 0.05,
    "w_loc": 0.02,
}


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


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
    if not bool(mask.any()):
        return _zero_loss(prediction)
    prediction = canonicalize_directions(prediction[mask])
    target = canonicalize_directions(target[mask])
    return F.mse_loss(prediction, target)


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any()):
        return _zero_loss(prediction)
    return F.mse_loss(prediction[mask], target[mask])


def direct_constraint_loss(
    predictions: Mapping[str, torch.Tensor],
    pmt_gt: torch.Tensor,
    mad_gt: torch.Tensor,
    dim_gt: torch.Tensor,
    loc_gt: torch.Tensor,
    *,
    weights: Mapping[str, float] | None = None,
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

    resolved_weights = dict(DEFAULT_DIRECT_LOSS_WEIGHTS)
    if weights is not None:
        unknown = sorted(set(weights).difference(resolved_weights))
        if unknown:
            raise ValueError(f"unknown direct loss weights: {unknown}")
        resolved_weights.update({name: float(value) for name, value in weights.items()})
    negative = {
        name: value for name, value in resolved_weights.items() if value < 0.0
    }
    if negative:
        raise ValueError(f"direct loss weights must be non-negative: {negative}")

    pmt_loss = F.nll_loss(log_pmt.reshape(-1, 5), pmt_gt.reshape(-1).long())
    mad_mask = _primitive_mask(pmt_gt, (0, 1, 2))
    dim_mask = _primitive_mask(pmt_gt, (1, 2, 3))
    loc_mask = _primitive_mask(pmt_gt, (0, 1, 2, 3))
    mad_loss = _masked_direction_mse(mad_pred, mad_gt, mad_mask)
    dim_loss = _masked_mse(dim_pred, dim_gt, dim_mask)
    loc_loss = _masked_mse(loc_pred, loc_gt, loc_mask)

    raw_losses = {
        "pmt": pmt_loss,
        "mad": mad_loss,
        "dim": dim_loss,
        "loc": loc_loss,
    }
    weighted_losses = {
        name: raw_losses[name] * resolved_weights[f"w_{name}"]
        for name in DIRECT_LOSS_NAMES
    }
    total = sum(weighted_losses.values(), _zero_loss(log_pmt))
    loss_dict: dict[str, torch.Tensor] = {
        "loss_all": total,
        "pmt_loss": pmt_loss,
        "mad_loss": mad_loss,
        "dim_loss": dim_loss,
        "loc_loss": loc_loss,
    }
    for name in DIRECT_LOSS_NAMES:
        loss_dict[f"raw/{name}"] = raw_losses[name]
        loss_dict[f"weighted/{name}"] = weighted_losses[name]
        loss_dict[f"effective_weight/{name}"] = total.new_tensor(
            resolved_weights[f"w_{name}"]
        )

    non_finite = [
        name
        for name, value in loss_dict.items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]
    if non_finite:
        raise FloatingPointError(f"non-finite Stage 1 direct losses: {non_finite}")
    return total, loss_dict


__all__ = [
    "DEFAULT_DIRECT_LOSS_WEIGHTS",
    "DIRECT_LOSS_NAMES",
    "direct_constraint_loss",
]
