"""Standalone full-dataset evaluator for Stage 1 direct baselines."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from functional.stage1_direct_loss import direct_constraint_loss
from functional.stage1_direct_metrics import Stage1DirectMetricAccumulator


class Stage1DirectEvaluator:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        data_loader: Any,
        device: torch.device,
        loss_weights: Mapping[str, float],
        use_amp: bool = False,
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.loss_weights = dict(loss_weights)
        self.use_amp = bool(use_amp and device.type == "cuda")

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(dtype=torch.float16)

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        self.model.eval()
        metrics = Stage1DirectMetricAccumulator()
        loss_totals: dict[str, float] = {}
        sample_count = 0
        for batch in tqdm(self.data_loader, desc="evaluate Stage 1 direct"):
            xyz = batch[0].float().to(self.device, non_blocking=True)
            pmt_gt = batch[1].long().to(self.device, non_blocking=True)
            mad_gt = batch[2].float().to(self.device, non_blocking=True)
            dim_gt = batch[3].float().to(self.device, non_blocking=True)
            loc_gt = batch[4].float().to(self.device, non_blocking=True)
            with self._autocast():
                predictions = self.model(xyz)
                _, loss_dict = direct_constraint_loss(
                    predictions,
                    pmt_gt,
                    mad_gt,
                    dim_gt,
                    loc_gt,
                    weights=self.loss_weights,
                )
            batch_size = int(xyz.shape[0])
            sample_count += batch_size
            for name, value in loss_dict.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    loss_totals[name] = loss_totals.get(name, 0.0) + (
                        float(value.detach().cpu()) * batch_size
                    )
            metrics.update(predictions, pmt_gt, mad_gt, dim_gt, loc_gt)
        summary = metrics.compute()
        summary["loss"] = {
            name: total / max(sample_count, 1)
            for name, total in loss_totals.items()
        }
        summary["sample_count"] = sample_count
        return summary

    @staticmethod
    def save(summary: Mapping[str, Any], path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(dict(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


__all__ = ["Stage1DirectEvaluator"]
