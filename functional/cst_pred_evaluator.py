from __future__ import annotations

import torch
from colorama import Fore
from tqdm import tqdm

from functional.direct_constraints import direct_constraints
from functional.cst_pred_trainer import (
    LOSS_NAMES,
    _aggregate_metric_dicts,
    _detach_dict,
    _mean_dicts,
    _scalar,
    _to_python,
    warn_if_primitive_collapsed,
)
from functional.stage1_phase_loss import stage1_phase_loss
from functional.point_features import stage1_forward
from functional.finite_checks import assert_finite_tensors
from functional.stage1_metrics import (
    aggregate_constraint_attribute_metrics,
    evaluate_constraint_attribute_metrics,
    evaluate_primitive_metrics,
    primitive_metrics_from_confusion,
)


class CstPredEvaluator:
    """Evaluate a frozen Stage 1 checkpoint over one complete TXT directory."""

    def __init__(
        self,
        model,
        data_loader,
        *,
        loss_weights,
        train_phase,
        use_extra_features,
        feature_k,
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        self.loss_weights = dict(loss_weights)
        self.train_phase = train_phase
        self.use_extra_features = bool(use_extra_features)
        self.feature_k = int(feature_k)

    @torch.no_grad()
    def evaluate(self, global_epoch):
        self.model.eval()
        loss_batches = []
        metric_batches = []
        direct_metric_batches = []
        progress = tqdm(self.data_loader, desc="evaluate Stage 1")
        for data_batch in progress:
            loss_dict, metric_dict, direct_metric_dict = self._process_batch(
                data_batch,
                global_epoch=global_epoch,
            )
            progress.set_postfix({
                "loss": f"{_scalar(loss_dict, 'loss_all'):.4f}",
                "pmt_acc": f"{_scalar(metric_dict, 'pmt_acc'):.4f}",
            })
            loss_batches.append(_detach_dict(loss_dict))
            metric_batches.append(_detach_dict(metric_dict))
            direct_metric_batches.append(_detach_dict(direct_metric_dict))

        if not loss_batches:
            raise ValueError("evaluation dataset produced no batches")

        loss_summary = _mean_dicts(loss_batches)
        metric_summary = _aggregate_metric_dicts(metric_batches)
        metric_summary.update(self._aggregate_direct_metrics(direct_metric_batches))
        metric_summary["constraint_score"] = -float(loss_summary["loss_all"])
        warn_if_primitive_collapsed(
            metric_summary,
            split="eval",
            epoch=global_epoch,
        )
        self._print_summary(loss_summary, metric_summary)
        return loss_summary, metric_summary

    def _process_batch(self, data_batch, *, global_epoch):
        """Stage1ConstraintDataset order: xyz, pmt, mad, dim, loc, affiliate_idx."""
        xyz = data_batch[0].float().to(self.device, non_blocking=True)
        pmt_gt = data_batch[1].long().to(self.device, non_blocking=True)
        mad_gt = data_batch[2].float().to(self.device, non_blocking=True)
        dim_gt = data_batch[3].float().to(self.device, non_blocking=True)
        loc_gt = data_batch[4].float().to(self.device, non_blocking=True)
        affiliate_idx = data_batch[-1].long().to(self.device, non_blocking=True)

        outputs = stage1_forward(self.model, xyz, use_extra_features=self.use_extra_features, feature_k=self.feature_k)
        self._validate_outputs(outputs)
        _, loss_dict = stage1_phase_loss(
            {name: value.float() for name, value in outputs.items()},
            pmt_gt, mad_gt, dim_gt, loc_gt, affiliate_idx,
            train_phase=self.train_phase, weights=self.loss_weights,
        )
        self._validate_losses(loss_dict)

        constraints = direct_constraints(outputs)
        metric_dict = {}
        metric_dict.update(evaluate_primitive_metrics(outputs["log_pmt"], pmt_gt))
        metric_dict.update(evaluate_constraint_attribute_metrics(
            mad_pred=outputs["mad"],
            dim_pred=outputs["dim"],
            loc_pred=outputs["loc"],
            pmt_gt=pmt_gt,
            mad_gt=mad_gt,
            dim_gt=dim_gt,
            loc_gt=loc_gt,
        ))
        direct_metric_dict = {
            "pmt_confusion_matrix": evaluate_primitive_metrics(
                constraints["primitive_type"], pmt_gt
            )["pmt_confusion_matrix"]
        }
        direct_metric_dict.update(evaluate_constraint_attribute_metrics(
            mad_pred=constraints["direction"],
            dim_pred=constraints["dimension"],
            loc_pred=constraints["location"],
            pmt_gt=pmt_gt,
            mad_gt=mad_gt,
            dim_gt=dim_gt,
            loc_gt=loc_gt,
        ))

        aggregation_weight = torch.tensor(
            float(xyz.shape[0]),
            device=xyz.device,
            dtype=torch.float32,
        )
        loss_dict["_aggregation_weight"] = aggregation_weight
        metric_dict["_aggregation_weight"] = aggregation_weight
        return loss_dict, metric_dict, direct_metric_dict

    @staticmethod
    def _aggregate_direct_metrics(metric_batches):
        confusion = torch.stack([
            batch["pmt_confusion_matrix"].float() for batch in metric_batches
        ]).sum(dim=0)
        primitive = primitive_metrics_from_confusion(confusion)
        attributes = aggregate_constraint_attribute_metrics(metric_batches)
        output = {
            f"direct_{name}": value
            for name, value in _to_python(primitive).items()
        }
        output.update({
            f"direct_{name}": value for name, value in attributes.items()
        })
        return output

    @staticmethod
    def _validate_outputs(outputs):
        if not isinstance(outputs, dict):
            raise TypeError("Stage 1 model must return a multitask prediction dictionary")
        required = {"embedding", "log_pmt", "mad", "dim", "loc"}
        missing = sorted(required.difference(outputs))
        if missing:
            raise ValueError(f"Stage 1 model output is missing fields: {missing}")
        assert_finite_tensors(outputs, "Stage 1 outputs")

    @staticmethod
    def _validate_losses(loss_dict):
        assert_finite_tensors(loss_dict, "Stage 1 losses")

    @staticmethod
    def _print_summary(loss_summary, metric_summary):
        raw_text = ", ".join(
            f"{name}={float(loss_summary.get('raw/' + name, 0.0)):.5f}"
            for name in LOSS_NAMES if f"raw/{name}" in loss_summary
        )
        print(
            Fore.CYAN
            + f"eval: loss_all={loss_summary.get('loss_all', 0.0):.6f}; "
            + f"raw[{raw_text}]"
        )
        print(
            Fore.CYAN
            + f"eval: pmt_acc={metric_summary.get('pmt_acc', 0.0):.4f}, "
            + f"pmt_macro_f1={metric_summary.get('pmt_macro_f1', 0.0):.4f}, "
            + f"pmt_miou={metric_summary.get('pmt_miou', 0.0):.4f}"
        )
        print(
            "eval: predicted primitive histogram="
            f"{metric_summary.get('pmt_pred_histogram', [])}"
        )
        print(
            "eval: confusion matrix="
            f"{metric_summary.get('pmt_confusion_matrix', [])}"
        )
        print(
            Fore.CYAN
            + "eval direct pipeline: "
            + f"pmt_acc={metric_summary.get('direct_pmt_acc', 0.0):.4f}, "
            + "direction="
            + f"{metric_summary.get('direct_direction_mean_angular_error_deg', 0.0):.4f}deg, "
            + "dimension="
            + f"{metric_summary.get('direct_dimension_mean_absolute_error', 0.0):.6f}, "
            + "location="
            + f"{metric_summary.get('direct_location_mean_distance_error', 0.0):.6f}"
        )
