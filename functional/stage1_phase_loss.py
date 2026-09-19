"""Five direct losses with a fixed semantic / geometry / joint schedule."""

import math

import torch
import torch.nn.functional as F

from functional.loss import discriminative_loss
from functional.stage1_direct_loss import (
    _masked_direction_mse,
    _masked_mse,
    _attribute_mask,
)


TRAINING_RECIPE = "simple_five_losses_v1"
LOSS_NAMES = ("pmt", "cluster", "mad", "dim", "loc")


def stage1_active_losses(train_phase):
    if train_phase not in ("semantic", "geometry", "joint"):
        raise ValueError(f"unsupported Stage 1 train phase: {train_phase}")
    return {
        name: train_phase == "joint" or (
            train_phase == "semantic" if name in ("pmt", "cluster")
            else train_phase == "geometry"
        )
        for name in LOSS_NAMES
    }


def stage1_phase_loss(predictions, pmt_gt, mad_gt, dim_gt, loc_gt,
                      affiliate_idx, *, train_phase, weights=None,
                      mad_valid_mask=None, dim_valid_mask=None, loc_valid_mask=None):
    """Direct point means only; inactive terms are never evaluated."""
    weights = {} if weights is None else weights
    active = stage1_active_losses(train_phase)
    raw = {}
    if active["pmt"]:
        raw["pmt"] = F.nll_loss(
            predictions["log_pmt"].reshape(-1, 5), pmt_gt.reshape(-1).long()
        )
    if active["cluster"]:
        raw["cluster"] = discriminative_loss(predictions["embedding"], affiliate_idx)
    if active["mad"]:
        raw["mad"] = _masked_direction_mse(
            predictions["mad"], mad_gt, _attribute_mask(pmt_gt, (0, 1, 2), mad_valid_mask)
        )
    if active["dim"]:
        raw["dim"] = _masked_mse(
            predictions["dim"], dim_gt, _attribute_mask(pmt_gt, (1, 2, 3), dim_valid_mask)
        )
    if active["loc"]:
        raw["loc"] = _masked_mse(
            predictions["loc"], loc_gt, _attribute_mask(pmt_gt, (0, 1, 2, 3), loc_valid_mask)
        )
    logs = {}
    total = pmt_gt.new_zeros((), dtype=torch.float32)
    for name, value in raw.items():
        weight = float(weights.get("w_" + name, 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"w_{name} must be finite and positive")
        weighted = value * weight
        total = total + weighted
        logs[f"raw/{name}"] = value
        logs[f"weighted/{name}"] = weighted
        logs[f"effective_weight/{name}"] = value.new_tensor(weight)
    logs["loss_all"] = total
    return total, logs
