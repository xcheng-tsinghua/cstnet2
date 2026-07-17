from __future__ import annotations

import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bars are optional
    def tqdm(iterable, **_kwargs):
        return iterable

from functional.segmentation_loss import WeightedSegmentationLoss
from functional.segmentation_metrics import SegmentationMetrics


class ContinuousWarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Per-step linear warmup followed by a continuous cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        super().__init__(optimizer, last_epoch=last_epoch)

    def _factor(self, step: int) -> float:
        step = max(int(step), 0)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        decay_steps = max(self.total_steps - self.warmup_steps - 1, 1)
        progress = min(max(step - self.warmup_steps, 0) / decay_steps, 1.0)
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    def get_lr(self) -> list[float]:
        factor = self._factor(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]


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


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _distributed_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _load_full_checkpoint(path: str | os.PathLike[str]):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument was introduced.
        return torch.load(path, map_location="cpu")


class Stage2SegmentationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Any,
        val_loader: Any,
        class_weights: torch.Tensor,
        label_map: dict[str, Any],
        output_dir: str | os.PathLike[str],
        device: torch.device,
        epochs: int = 200,
        warmup_epochs: int = 5,
        gradient_clip_norm: float = 1.0,
        use_amp: bool = True,
        checkpoint_args: dict[str, Any] | None = None,
        wandb_run: Any = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = int(epochs)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.label_map = label_map
        self.num_classes = len(label_map["labels"])
        self.output_dir = Path(output_dir)
        self.checkpoint_args = dict(checkpoint_args or {})
        self.wandb_run = wandb_run
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.is_main = self.rank == 0
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32).cpu()
        self.criterion = WeightedSegmentationLoss(self.class_weights).to(device)
        total_steps = self.epochs * max(len(train_loader), 1)
        warmup_steps = int(warmup_epochs) * max(len(train_loader), 1)
        self.scheduler = ContinuousWarmupCosineLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        self.use_amp = bool(use_amp and device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):  # PyTorch < 2.3
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = float("-inf")

    @property
    def unwrapped_model(self) -> torch.nn.Module:
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def _autocast(self):
        if self.use_amp:
            try:
                return torch.amp.autocast("cuda", dtype=torch.float16)
            except (AttributeError, TypeError):  # PyTorch < 2.0
                return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()

    def _batch_to_device(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            key: batch[key].to(self.device, non_blocking=True)
            for key in ("xyz", "constraints", "constraint_masks", "labels", "face_ids")
        }

    def _backward_and_step(self, loss: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Backpropagate once and report whether AMP skipped the optimizer step."""
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        if not self.use_amp and not bool(
            torch.isfinite(torch.as_tensor(gradient_norm)).item()
        ):
            raise FloatingPointError("Stage 2 segmentation gradient norm is NaN or Inf")

        scale_before_step = float(self.scaler.get_scale())
        # With AMP enabled, GradScaler skips optimizer.step() when unscale_ found
        # Inf/NaN gradients and lowers the scale so a later batch can recover.
        self.scaler.step(self.optimizer)
        self.scaler.update()
        optimizer_step_skipped = (
            self.use_amp and float(self.scaler.get_scale()) < scale_before_step
        )
        return torch.as_tensor(gradient_norm), optimizer_step_skipped

    def _run_epoch(self, loader: Any, training: bool, epoch: int) -> tuple[float, dict[str, Any]]:
        self.model.train(training)
        metrics = SegmentationMetrics(self.num_classes)
        total_loss = 0.0
        batch_count = 0
        amp_skipped_steps = 0
        description = f"train {epoch + 1}/{self.epochs}" if training else f"val {epoch + 1}/{self.epochs}"
        iterator = tqdm(loader, desc=description, disable=not self.is_main)

        context = nullcontext() if training else torch.no_grad()
        with context:
            for raw_batch in iterator:
                batch = self._batch_to_device(raw_batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits = self.model(
                        batch["xyz"],
                        batch["constraints"],
                        batch["constraint_masks"],
                    )
                    loss = self.criterion(logits, batch["labels"])

                if training:
                    _, optimizer_step_skipped = self._backward_and_step(loss)
                    if optimizer_step_skipped:
                        amp_skipped_steps += 1
                    else:
                        self.scheduler.step()
                        self.global_step += 1

                total_loss += float(loss.detach().item())
                batch_count += 1
                metrics.update(logits.detach(), batch["labels"], batch["face_ids"])
                if self.is_main and hasattr(iterator, "set_postfix"):
                    postfix = {"loss": f"{loss.item():.4f}"}
                    if amp_skipped_steps:
                        postfix["amp_skips"] = amp_skipped_steps
                    iterator.set_postfix(**postfix)

        total_loss = _distributed_sum(total_loss, self.device)
        batch_count = int(_distributed_sum(float(batch_count), self.device))
        return total_loss / max(batch_count, 1), metrics.compute()

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model": self.unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "best_metric": float(self.best_metric),
            "args": self.checkpoint_args,
            "label_map": self.label_map,
            "class_weights": self.class_weights,
            "rng_state": capture_rng_state(),
        }

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        if not self.is_main:
            return
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        torch.save(self._checkpoint_payload(epoch), temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        checkpoint = _load_full_checkpoint(checkpoint_path)
        required = {
            "epoch", "global_step", "model", "optimizer", "scheduler", "scaler",
            "best_metric", "args", "label_map", "class_weights", "rng_state",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ValueError(f"incomplete Stage 2 segmentation checkpoint; missing: {missing}")
        if checkpoint["label_map"] != self.label_map:
            raise ValueError("checkpoint label map does not match the current dataset metadata")
        saved_weights = torch.as_tensor(checkpoint["class_weights"], dtype=torch.float32).cpu()
        if saved_weights.shape != self.class_weights.shape or not torch.allclose(
            saved_weights, self.class_weights, rtol=1e-5, atol=1e-7
        ):
            raise ValueError("checkpoint class weights do not match training-only statistics")

        self.unwrapped_model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.use_amp:
            if checkpoint["scaler"] is None:
                raise ValueError("AMP checkpoint is missing GradScaler state")
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_metric = float(checkpoint["best_metric"])
        restore_rng_state(checkpoint["rng_state"])

    def fit(self, resume_checkpoint: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        if resume_checkpoint:
            checkpoint_path = Path(resume_checkpoint).expanduser().resolve()
            if self.is_main:
                print(f"checkpoint: loading {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
            if self.is_main:
                print(
                    f"checkpoint: loaded successfully; resuming at epoch "
                    f"{self.start_epoch + 1}/{self.epochs}, global_step={self.global_step}, "
                    f"best_point_mIoU={self.best_metric:.4f}"
                )
        elif self.is_main:
            print("checkpoint: none provided; starting a new training run at epoch 1")
        latest_metrics: dict[str, Any] = {}

        for epoch in range(self.start_epoch, self.epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_loss, train_metrics = self._run_epoch(self.train_loader, training=True, epoch=epoch)
            val_loss, val_metrics = self._run_epoch(self.val_loader, training=False, epoch=epoch)
            point_miou = float(val_metrics["point_mean_iou"])
            improved = point_miou > self.best_metric
            if improved:
                self.best_metric = point_miou

            self._save_checkpoint(self.output_dir / "last.pth", epoch)
            if improved:
                self._save_checkpoint(self.output_dir / "best_point_miou.pth", epoch)

            latest_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train": train_metrics,
                "val": val_metrics,
                "best_point_miou": self.best_metric,
            }
            if self.is_main:
                print(
                    f"epoch {epoch + 1}/{self.epochs} train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} point_mIoU={point_miou:.4f} "
                    f"face_mIoU={val_metrics['face_mean_iou']:.4f} best={self.best_metric:.4f}"
                )
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "loss/train": train_loss,
                            "loss/val": val_loss,
                            "point_miou/train": train_metrics["point_mean_iou"],
                            "point_miou/val": val_metrics["point_mean_iou"],
                            "face_miou/train": train_metrics["face_mean_iou"],
                            "face_miou/val": val_metrics["face_mean_iou"],
                            "point_accuracy/val": val_metrics["point_overall_accuracy"],
                            "face_accuracy/val": val_metrics["face_overall_accuracy"],
                        },
                        step=self.global_step,
                    )
        return latest_metrics
