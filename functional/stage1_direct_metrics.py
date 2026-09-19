"""Exact epoch-level metrics shared by direct baseline training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from functional.stage1_metrics import (
    CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS,
    TRIMMED_ATTRIBUTE_ACCUMULATOR_KEYS,
    aggregate_constraint_attribute_metrics,
    evaluate_constraint_attribute_metrics,
    primitive_metrics_from_confusion,
)
from networks.stage1_direct_baselines import finalize_direct_constraints


def _to_python(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    return value


class Stage1DirectMetricAccumulator:
    """Accumulate confusion counts and geometry-error sums without batch bias."""

    def __init__(self):
        self.confusion = torch.zeros(5, 5, dtype=torch.float64)
        self.raw_attributes = {
            key: 0.0 for key in CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS
        }
        self.final_attributes = {
            key: 0.0 for key in
            CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS | TRIMMED_ATTRIBUTE_ACCUMULATOR_KEYS
        }
        self.last_pmt_acc = torch.zeros(())

    @torch.no_grad()
    def update(
        self,
        predictions: Mapping[str, torch.Tensor],
        pmt_gt: torch.Tensor,
        mad_gt: torch.Tensor,
        dim_gt: torch.Tensor,
        loc_gt: torch.Tensor,
    ) -> None:
        pred = predictions["log_pmt"].detach().argmax(dim=-1).long()
        confusion = torch.bincount(
            (pmt_gt.long() * 5 + pred).reshape(-1), minlength=25
        ).reshape(5, 5)
        self.confusion = self.confusion.to(confusion.device)
        self.confusion.add_(confusion)
        self.last_pmt_acc = confusion.diag().sum().float() / max(pmt_gt.numel(), 1)

        raw = evaluate_constraint_attribute_metrics(
            predictions["mad"],
            predictions["dim"],
            predictions["loc"],
            pmt_gt,
            mad_gt,
            dim_gt,
            loc_gt,
        )
        finalized = finalize_direct_constraints(predictions)
        final = evaluate_constraint_attribute_metrics(
            finalized["direction"],
            finalized["dimension"],
            finalized["location"],
            pmt_gt,
            mad_gt,
            dim_gt,
            loc_gt,
            include_trimmed=True,
        )
        for key in CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS:
            self.raw_attributes[key] = self.raw_attributes[key] + raw[key].detach().double()
        for key in self.final_attributes:
            self.final_attributes[key] = self.final_attributes[key] + final[key].detach().double()

    def compute(self) -> dict[str, Any]:
        keys = sorted(CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS)
        final_keys = sorted(self.final_attributes)
        packed = torch.stack([
            torch.as_tensor(values[key], device=self.confusion.device, dtype=torch.float64)
            for values, group_keys in ((self.raw_attributes, keys), (self.final_attributes, final_keys))
            for key in group_keys
        ])
        # One transfer at epoch end; keep the accumulator reusable after compute().
        host = torch.cat((self.confusion.reshape(-1), packed)).cpu()
        primitive = primitive_metrics_from_confusion(host[:25].reshape(5, 5).float())
        raw = aggregate_constraint_attribute_metrics([dict(zip(keys, host[25:25 + len(keys)]))])
        final = aggregate_constraint_attribute_metrics([dict(zip(final_keys, host[25 + len(keys):]))])
        output = {str(key): _to_python(value) for key, value in primitive.items()}
        output.update({
            key: value for key, value in raw.items()
            if not key.endswith("_valid_points")
        })
        output.update({
            (key if key.startswith("trim") else f"final/{key}"): value
            for key, value in final.items()
            if not key.endswith("_valid_points")
        })
        return output


__all__ = ["Stage1DirectMetricAccumulator"]
