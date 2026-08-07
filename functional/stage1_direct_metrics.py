"""Exact epoch-level metrics shared by direct baseline training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from functional.stage1_metrics import (
    CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS,
    aggregate_constraint_attribute_metrics,
    evaluate_constraint_attribute_metrics,
    evaluate_primitive_metrics,
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
            key: 0.0 for key in CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS
        }

    @torch.no_grad()
    def update(
        self,
        predictions: Mapping[str, torch.Tensor],
        pmt_gt: torch.Tensor,
        mad_gt: torch.Tensor,
        dim_gt: torch.Tensor,
        loc_gt: torch.Tensor,
    ) -> None:
        primitive = evaluate_primitive_metrics(predictions["log_pmt"], pmt_gt)
        self.confusion += primitive["pmt_confusion_matrix"].detach().double().cpu()

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
        )
        for key in CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS:
            self.raw_attributes[key] += float(raw[key].detach().cpu())
            self.final_attributes[key] += float(final[key].detach().cpu())

    def compute(self) -> dict[str, Any]:
        primitive = primitive_metrics_from_confusion(self.confusion.float())
        raw = aggregate_constraint_attribute_metrics([self.raw_attributes])
        final = aggregate_constraint_attribute_metrics([self.final_attributes])
        output = {str(key): _to_python(value) for key, value in primitive.items()}
        output.update(raw)
        output.update({f"final/{key}": value for key, value in final.items()})
        return output


__all__ = ["Stage1DirectMetricAccumulator"]
