"""Independent trainer for XYZ-only Stage 1 direct baselines."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from functional.checkpoint_io import safe_torch_save
from functional.stage1_direct_loss import direct_constraint_loss
from functional.stage1_direct_metrics import Stage1DirectMetricAccumulator
from functional.stage1_metrics import primitive_prediction_collapsed
from functional.wandb_utils import (
    flatten_wandb_summary_metrics,
    wandb_confusion_matrix,
    wandb_run_id,
)
from networks.stage1_direct_baselines import stage1_direct_model_config


PRIMITIVE_CLASS_NAMES = ("plane", "cylinder", "cone", "sphere", "other")
DIRECT_CHECKPOINT_TASK = "stage1_direct_baseline"
DIRECT_CHECKPOINT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_direct_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage 1 direct checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # PyTorch before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid Stage 1 direct checkpoint: {checkpoint_path}")
    return checkpoint


def _scalar_loss_summary(
    totals: Mapping[str, float], total_samples: int
) -> dict[str, float]:
    denominator = max(int(total_samples), 1)
    return {name: value / denominator for name, value in totals.items()}


class Stage1DirectTrainer:
    """Train direct baselines without importing the existing Stage 1 trainer."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: Any,
        val_loader: Any | None,
        output_dir: str | os.PathLike[str],
        device: torch.device,
        epochs: int,
        loss_weights: Mapping[str, float],
        gradient_clip_norm: float = 1.0,
        use_amp: bool = False,
        checkpoint_args: Mapping[str, Any] | None = None,
        wandb_run: Any = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.epochs = int(epochs)
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        self.loss_weights = {name: float(value) for name, value in loss_weights.items()}
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.use_amp = bool(use_amp and device.type == "cuda")
        if use_amp and not self.use_amp:
            print("WARNING: AMP requested without CUDA; AMP is disabled")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):  # PyTorch < 2.3
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.checkpoint_args = dict(checkpoint_args or {})
        self.model_config = stage1_direct_model_config(
            getattr(model, "model_config", self.checkpoint_args)
        )
        self.wandb_run = wandb_run
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.best_pmt_miou = float("-inf")
        self.history: list[dict[str, Any]] = []

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except (AttributeError, TypeError):  # PyTorch < 2.0
            return torch.cuda.amp.autocast(dtype=torch.float16)

    def _batch_to_device(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 5:
            raise TypeError(
                "Stage1ConstraintDataset batch must be "
                "(xyz, pmt, direction, dimension, location, affiliate_idx)"
            )
        return (
            batch[0].float().to(self.device, non_blocking=True),
            batch[1].long().to(self.device, non_blocking=True),
            batch[2].float().to(self.device, non_blocking=True),
            batch[3].float().to(self.device, non_blocking=True),
            batch[4].float().to(self.device, non_blocking=True),
        )

    @staticmethod
    def _assert_finite_predictions(predictions: Mapping[str, torch.Tensor]) -> None:
        required = {"log_pmt", "mad", "dim", "loc"}
        missing = sorted(required.difference(predictions))
        if missing:
            raise ValueError(f"direct model output is missing fields: {missing}")
        non_finite = [
            name
            for name, value in predictions.items()
            if torch.is_tensor(value) and not torch.isfinite(value).all()
        ]
        if non_finite:
            raise FloatingPointError(
                f"non-finite Stage 1 direct predictions: {non_finite}"
            )

    def _backward_and_step(self, loss: torch.Tensor) -> tuple[float, bool]:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        gradient_norm_value = float(torch.as_tensor(gradient_norm).detach().cpu())
        if not np.isfinite(gradient_norm_value) and not self.use_amp:
            raise FloatingPointError("Stage 1 direct gradient norm is NaN or Inf")
        scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.use_amp and float(self.scaler.get_scale()) < scale_before
        if not np.isfinite(gradient_norm_value) and not skipped:
            raise FloatingPointError(
                "Stage 1 direct gradient norm is NaN or Inf and the optimizer step was not skipped"
            )
        if not skipped:
            self.global_step += 1
        return gradient_norm_value, skipped

    def _run_epoch(
        self, loader: Any, *, training: bool, epoch: int
    ) -> tuple[dict[str, float], dict[str, Any]]:
        self.model.train(training)
        metrics = Stage1DirectMetricAccumulator()
        loss_totals: dict[str, float] = {}
        sample_count = 0
        gradient_sum = 0.0
        gradient_max = 0.0
        gradient_count = 0
        amp_skips = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start_time = perf_counter()
        mode = "train" if training else "val"
        iterator = tqdm(
            loader,
            desc=f"{mode} {epoch + 1}/{self.epochs}",
        )
        grad_context = nullcontext() if training else torch.no_grad()
        with grad_context:
            for raw_batch in iterator:
                xyz, pmt_gt, mad_gt, dim_gt, loc_gt = self._batch_to_device(raw_batch)
                batch_size = int(xyz.shape[0])
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    predictions = self.model(xyz)
                    self._assert_finite_predictions(predictions)
                    loss, loss_dict = direct_constraint_loss(
                        predictions,
                        pmt_gt,
                        mad_gt,
                        dim_gt,
                        loc_gt,
                        weights=self.loss_weights,
                    )

                if training:
                    gradient_norm, skipped = self._backward_and_step(loss)
                    if skipped:
                        amp_skips += 1
                    else:
                        gradient_sum += gradient_norm
                        gradient_max = max(gradient_max, gradient_norm)
                        gradient_count += 1

                for name, value in loss_dict.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        loss_totals[name] = loss_totals.get(name, 0.0) + (
                            float(value.detach().cpu()) * batch_size
                        )
                sample_count += batch_size
                metrics.update(
                    predictions, pmt_gt, mad_gt, dim_gt, loc_gt
                )
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        loss=f"{float(loss.detach().cpu()):.4f}",
                        pmt=f"{float((predictions['log_pmt'].argmax(-1) == pmt_gt).float().mean()):.3f}",
                    )

        elapsed = max(perf_counter() - start_time, 1e-9)
        loss_summary = _scalar_loss_summary(loss_totals, sample_count)
        metric_summary = metrics.compute()
        metric_summary.update(
            {
                "efficiency/samples_per_second": sample_count / elapsed,
                "efficiency/epoch_seconds": elapsed,
                "efficiency/peak_memory_mb": (
                    torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
                    if self.device.type == "cuda"
                    else 0.0
                ),
            }
        )
        if training:
            metric_summary.update(
                {
                    "optimization/gradient_norm_mean": gradient_sum
                    / max(gradient_count, 1),
                    "optimization/gradient_norm_max": gradient_max,
                    "optimization/amp_skipped_steps": amp_skips,
                    "optimization/amp_scale": float(self.scaler.get_scale()),
                }
            )
        if primitive_prediction_collapsed(
            torch.as_tensor(metric_summary["pmt_pred_histogram"])
        ):
            print(
                f"WARNING: {mode} primitive prediction collapsed at epoch {epoch + 1}: "
                f"{metric_summary['pmt_pred_histogram']}"
            )
        return loss_summary, metric_summary

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "task": DIRECT_CHECKPOINT_TASK,
            "format_version": DIRECT_CHECKPOINT_VERSION,
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "best_loss": float(self.best_loss),
            "best_pmt_miou": float(self.best_pmt_miou),
            "args": self.checkpoint_args,
            "model_config": self.model_config,
            "loss_weights": self.loss_weights,
            "rng_state": capture_rng_state(),
            "wandb_run_id": wandb_run_id(self.wandb_run),
        }

    def _save_checkpoint(self, filename: str, epoch: int) -> bool:
        return safe_torch_save(
            self._checkpoint_payload(epoch), self.output_dir / filename
        )

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        checkpoint = load_direct_checkpoint(path)
        required = {
            "task",
            "epoch",
            "global_step",
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "best_loss",
            "best_pmt_miou",
            "model_config",
            "loss_weights",
            "rng_state",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ValueError(f"incomplete Stage 1 direct checkpoint; missing: {missing}")
        if checkpoint["task"] != DIRECT_CHECKPOINT_TASK:
            raise ValueError(
                f"checkpoint task is {checkpoint['task']!r}, expected {DIRECT_CHECKPOINT_TASK!r}"
            )
        saved_config = stage1_direct_model_config(checkpoint["model_config"])
        if saved_config != self.model_config:
            raise ValueError(
                "Stage 1 direct checkpoint model configuration mismatch: "
                f"saved={saved_config}, requested={self.model_config}"
            )
        saved_weights = {
            name: float(value) for name, value in checkpoint["loss_weights"].items()
        }
        if saved_weights != self.loss_weights:
            raise ValueError(
                "Stage 1 direct checkpoint loss weights mismatch: "
                f"saved={saved_weights}, requested={self.loss_weights}"
            )

        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            if checkpoint["scheduler"] is None:
                raise ValueError("Stage 1 direct checkpoint is missing scheduler state")
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.use_amp:
            if checkpoint["scaler"] is None:
                raise ValueError("AMP resume checkpoint is missing GradScaler state")
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_loss = float(checkpoint["best_loss"])
        self.best_pmt_miou = float(checkpoint["best_pmt_miou"])
        restore_rng_state(checkpoint["rng_state"])

    def _write_history(self) -> None:
        destination = self.output_dir / "history.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def _log_wandb(self, record: Mapping[str, Any]) -> None:
        if self.wandb_run is None:
            return
        payload: dict[str, Any] = {
            "epoch": int(record["epoch"]) + 1,
            "global_step": self.global_step,
            "learning_rate": max(
                float(group["lr"]) for group in self.optimizer.param_groups
            ),
        }
        payload.update(flatten_wandb_summary_metrics("train/loss", record["train_loss"]))
        payload.update(flatten_wandb_summary_metrics("train", record["train"]))
        if record.get("val_loss") is not None:
            payload.update(flatten_wandb_summary_metrics("val/loss", record["val_loss"]))
            payload.update(flatten_wandb_summary_metrics("val", record["val"]))
        metric_source = record["val"] if record.get("val") is not None else record["train"]
        payload["primitive_confusion"] = wandb_confusion_matrix(
            metric_source["pmt_confusion_matrix"],
            PRIMITIVE_CLASS_NAMES,
            title="Stage 1 Direct Primitive Confusion",
        )
        self.wandb_run.log(payload, step=int(record["epoch"]) + 1)

    def fit(
        self, resume_checkpoint: str | os.PathLike[str] | None = None
    ) -> dict[str, Any]:
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
            history_path = self.output_dir / "history.json"
            if history_path.is_file():
                loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_history, list):
                    raise ValueError(f"invalid Stage 1 direct history: {history_path}")
                self.history = loaded_history
            print(
                f"resumed Stage 1 direct training from {resume_checkpoint}; "
                f"next_epoch={self.start_epoch + 1}, global_step={self.global_step}"
            )
        else:
            print("starting new Stage 1 direct baseline training")
        if self.val_loader is None:
            print(
                "WARNING: --val_data_root was not provided; best checkpoints use "
                "training metrics. Use a validation set for formal comparisons."
            )

        latest: dict[str, Any] = {}
        for epoch in range(self.start_epoch, self.epochs):
            train_loss, train_metrics = self._run_epoch(
                self.train_loader, training=True, epoch=epoch
            )
            if self.val_loader is None:
                val_loss, val_metrics = None, None
                selection_loss = float(train_loss["loss_all"])
                selection_miou = float(train_metrics["pmt_miou"])
            else:
                val_loss, val_metrics = self._run_epoch(
                    self.val_loader, training=False, epoch=epoch
                )
                selection_loss = float(val_loss["loss_all"])
                selection_miou = float(val_metrics["pmt_miou"])

            improved_loss = selection_loss < self.best_loss
            improved_miou = selection_miou > self.best_pmt_miou
            if improved_loss:
                self.best_loss = selection_loss
            if improved_miou:
                self.best_pmt_miou = selection_miou
            if self.scheduler is not None:
                self.scheduler.step()

            last_saved = self._save_checkpoint("last.pth", epoch)
            best_loss_saved = (
                self._save_checkpoint("best_loss.pth", epoch)
                if improved_loss
                else True
            )
            best_miou_saved = (
                self._save_checkpoint("best_pmt_miou.pth", epoch)
                if improved_miou
                else True
            )
            latest = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train": train_metrics,
                "val_loss": val_loss,
                "val": val_metrics,
                "best_loss": self.best_loss,
                "best_pmt_miou": self.best_pmt_miou,
                "checkpoint/last_saved": last_saved,
                "checkpoint/best_loss_saved": best_loss_saved,
                "checkpoint/best_pmt_miou_saved": best_miou_saved,
            }
            self.history.append(latest)
            self._write_history()
            source_metrics = val_metrics if val_metrics is not None else train_metrics
            print(
                f"epoch {epoch + 1}/{self.epochs} "
                f"train_loss={train_loss['loss_all']:.6f} "
                f"selection_loss={selection_loss:.6f} "
                f"pmt_mIoU={source_metrics['pmt_miou']:.4f} "
                f"direction={source_metrics['final/direction_mean_angular_error_deg']:.3f}deg "
                f"dimension={source_metrics['final/dimension_mean_absolute_error']:.6f} "
                f"location={source_metrics['final/location_mean_distance_error']:.6f}"
            )
            self._log_wandb(latest)
        return latest


__all__ = [
    "DIRECT_CHECKPOINT_TASK",
    "Stage1DirectTrainer",
    "capture_rng_state",
    "load_direct_checkpoint",
    "restore_rng_state",
]
