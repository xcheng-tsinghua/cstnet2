from __future__ import annotations

import torch
from colorama import Fore
from tqdm import tqdm

from functional.cst_pred_trainer import (
    LOSS_NAMES,
    _aggregate_metric_dicts,
    _detach_dict,
    _mean_dicts,
    _scalar,
    stage1_active_losses,
    warn_if_primitive_collapsed,
)
from functional.loss import constraint_loss, evaluate_clustering
from functional.point_features import build_stage1_input_features
from functional.stage1_metrics import (
    evaluate_constraint_attribute_metrics,
    evaluate_predicted_clustering,
    evaluate_primitive_metrics,
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
        enabled_losses,
        geom_start_epoch,
        geom_ramp_epochs,
        use_extra_features,
        feature_k,
        cluster_bandwidth,
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        self.loss_weights = dict(loss_weights)
        self.train_phase = train_phase
        self.enabled_losses = dict(enabled_losses)
        self.geom_start_epoch = int(geom_start_epoch)
        self.geom_ramp_epochs = int(geom_ramp_epochs)
        self.use_extra_features = bool(use_extra_features)
        self.feature_k = int(feature_k)
        self.cluster_bandwidth = float(cluster_bandwidth)

    @torch.no_grad()
    def evaluate(self, global_epoch):
        self.model.eval()
        loss_batches = []
        metric_batches = []
        active_losses = stage1_active_losses(
            self.train_phase,
            self.enabled_losses,
        )
        progress = tqdm(self.data_loader, desc="evaluate Stage 1")
        for data_batch in progress:
            loss_dict, metric_dict = self._process_batch(
                data_batch,
                global_epoch=global_epoch,
                active_losses=active_losses,
            )
            progress.set_postfix({
                "loss": f"{_scalar(loss_dict, 'loss_all'):.4f}",
                "pmt_acc": f"{_scalar(metric_dict, 'pmt_acc'):.4f}",
                "ari_real": f"{_scalar(metric_dict, 'cluster_ari_real'):.4f}",
            })
            loss_batches.append(_detach_dict(loss_dict))
            metric_batches.append(_detach_dict(metric_dict))

        if not loss_batches:
            raise ValueError("evaluation dataset produced no batches")

        loss_summary = _mean_dicts(loss_batches)
        metric_summary = _aggregate_metric_dicts(metric_batches)
        metric_summary["constraint_score"] = 0.5 * (
            float(metric_summary.get("pmt_miou", 0.0))
            + max(0.0, float(metric_summary.get("cluster_ari_real", 0.0)))
        )
        warn_if_primitive_collapsed(
            metric_summary,
            split="eval",
            epoch=global_epoch,
        )
        self._print_summary(loss_summary, metric_summary)
        return loss_summary, metric_summary

    def _process_batch(self, data_batch, *, global_epoch, active_losses):
        """Stage1ConstraintDataset order: xyz, pmt, mad, dim, loc, affiliate_idx."""
        xyz = data_batch[0].float().to(self.device, non_blocking=True)
        pmt_gt = data_batch[1].long().to(self.device, non_blocking=True)
        mad_gt = data_batch[2].float().to(self.device, non_blocking=True)
        dim_gt = data_batch[3].float().to(self.device, non_blocking=True)
        loc_gt = data_batch[4].float().to(self.device, non_blocking=True)
        affiliate_idx = data_batch[-1].long().to(self.device, non_blocking=True)

        extra_features = None
        if self.use_extra_features:
            extra_features = build_stage1_input_features(
                xyz,
                use_curvature=True,
                use_density=True,
                k=self.feature_k,
            )

        outputs = self.model(xyz, extra_features)
        self._validate_outputs(outputs)
        _, loss_dict = constraint_loss(
            xyz=xyz,
            log_pmt_pred=outputs["log_pmt"].float(),
            mad_pred=outputs["mad"].float(),
            dim_pred=outputs["dim"].float(),
            loc_pred=outputs["loc"].float(),
            pmt_gt=pmt_gt,
            mad_gt=mad_gt,
            dim_gt=dim_gt,
            loc_gt=loc_gt,
            affil_idx=affiliate_idx,
            point_emb=outputs["embedding"].float(),
            weights=self.loss_weights,
            global_epoch=global_epoch,
            geom_start_epoch=self.geom_start_epoch,
            geom_ramp_epochs=self.geom_ramp_epochs,
            enabled_losses=active_losses,
        )
        self._validate_losses(loss_dict)

        metric_dict = {}
        metric_dict.update(evaluate_primitive_metrics(outputs["log_pmt"], pmt_gt))
        metric_dict.update(evaluate_predicted_clustering(
            affiliate_idx,
            outputs["embedding"],
            bandwidth=self.cluster_bandwidth,
        ))
        oracle_acc, oracle_nmi, oracle_ari = evaluate_clustering(
            affiliate_idx,
            outputs["embedding"],
        )
        metric_dict.update({
            "cluster_acc_oracle_optional": oracle_acc,
            "cluster_nmi_oracle_optional": oracle_nmi,
            "cluster_ari_oracle_optional": oracle_ari,
        })
        metric_dict.update(evaluate_constraint_attribute_metrics(
            mad_pred=outputs["mad"],
            dim_pred=outputs["dim"],
            loc_pred=outputs["loc"],
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
        return loss_dict, metric_dict

    @staticmethod
    def _validate_outputs(outputs):
        if not isinstance(outputs, dict):
            raise TypeError("Stage 1 model must return a multitask prediction dictionary")
        required = {"embedding", "log_pmt", "mad", "dim", "loc"}
        missing = sorted(required.difference(outputs))
        if missing:
            raise ValueError(f"Stage 1 model output is missing fields: {missing}")
        non_finite = [
            name
            for name, value in outputs.items()
            if torch.is_tensor(value) and not torch.isfinite(value).all()
        ]
        if non_finite:
            raise FloatingPointError(f"non-finite Stage 1 outputs: {non_finite}")

    @staticmethod
    def _validate_losses(loss_dict):
        non_finite = [
            name
            for name, value in loss_dict.items()
            if torch.is_tensor(value) and not torch.isfinite(value).all()
        ]
        if non_finite:
            raise FloatingPointError(f"non-finite Stage 1 losses: {non_finite}")

    @staticmethod
    def _print_summary(loss_summary, metric_summary):
        raw_text = ", ".join(
            f"{name}={float(loss_summary.get('raw/' + name, 0.0)):.5f}"
            for name in LOSS_NAMES
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
            + f"pmt_miou={metric_summary.get('pmt_miou', 0.0):.4f}, "
            + f"cluster_ari_real={metric_summary.get('cluster_ari_real', 0.0):.4f}, "
            + f"cluster_nmi_real={metric_summary.get('cluster_nmi_real', 0.0):.4f}"
        )
        print(
            "eval: predicted primitive histogram="
            f"{metric_summary.get('pmt_pred_histogram', [])}"
        )
        print(
            "eval: confusion matrix="
            f"{metric_summary.get('pmt_confusion_matrix', [])}"
        )
