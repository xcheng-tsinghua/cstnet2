from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights_from_frequency(
    class_frequency: torch.Tensor,
    eps: float = 1e-12,
    max_weight: float = 5.0,
) -> torch.Tensor:
    frequency = torch.as_tensor(class_frequency, dtype=torch.float64)
    if frequency.ndim != 1 or frequency.numel() == 0:
        raise ValueError("class_frequency must be a non-empty one-dimensional tensor")
    if not torch.isfinite(frequency).all() or (frequency < 0).any():
        raise ValueError("class_frequency must contain finite non-negative values")
    weights = 1.0 / torch.sqrt(frequency + eps)
    weights = weights / weights.mean().clamp_min(eps)
    return weights.clamp(max=max_weight).float()


def compute_training_class_statistics(
    dataset: Any,
    output_path: str | os.PathLike[str],
    force: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Scan only the training split and cache exact point-label frequencies."""
    if getattr(dataset, "split", None) != "train":
        raise ValueError("class weights may only be computed from the training split")
    output_path = Path(output_path)
    if output_path.is_file() and not force:
        with output_path.open("r", encoding="utf-8") as handle:
            statistics = json.load(handle)
        if statistics.get("label_map") != dataset.label_map:
            raise ValueError(f"cached class statistics use a different label map: {output_path}")
        if int(statistics.get("num_samples", -1)) != len(dataset.files):
            raise ValueError(
                "cached class statistics do not match the current training split size; "
                "the dataset may have changed or finished downloading. Re-run with "
                "--recompute_class_statistics"
            )
        weights = torch.tensor(statistics["class_weights"], dtype=torch.float32)
        if weights.numel() != dataset.num_classes:
            raise ValueError(f"cached class statistics have the wrong class count: {output_path}")
        return weights, statistics

    counts = np.zeros(dataset.num_classes, dtype=np.int64)
    for file_id, path in enumerate(dataset.files):
        if dataset.use_npy_cache:
            array = dataset._load_array(path)
            labels = array[:, 16].astype(np.int64)
        else:
            labels = np.loadtxt(path, dtype=np.int64, usecols=(16,))
            labels = np.atleast_1d(labels)
        if np.any((labels < 0) | (labels >= dataset.num_classes)):
            raise ValueError(f"out-of-range segmentation label while scanning {path}")
        counts += np.bincount(labels, minlength=dataset.num_classes)
        if (file_id + 1) % 2000 == 0:
            print(f"class statistics: scanned {file_id + 1}/{len(dataset.files)} training samples")

    total = int(counts.sum())
    if total <= 0:
        raise ValueError("the MFCAD++ training split contains no labeled points")
    frequency = torch.from_numpy(counts.astype(np.float64)) / float(total)
    weights = class_weights_from_frequency(frequency)
    statistics = {
        "split": "train",
        "num_samples": len(dataset.files),
        "total_points": total,
        "class_counts": counts.tolist(),
        "class_frequency": frequency.tolist(),
        "class_weights": weights.tolist(),
        "label_map": dataset.label_map,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output_path)
    return weights, statistics


def weighted_segmentation_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"expected logits [B, N, C], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels must have shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise FloatingPointError("segmentation logits contain NaN or Inf")
    num_classes = logits.shape[-1]
    class_weights = torch.as_tensor(class_weights, device=logits.device, dtype=logits.dtype)
    if class_weights.shape != (num_classes,):
        raise ValueError(f"expected {num_classes} class weights, got {tuple(class_weights.shape)}")
    if not torch.isfinite(class_weights).all() or (class_weights < 0).any():
        raise ValueError("class weights must be finite and non-negative")

    labels = labels.long()
    valid = labels != ignore_index
    if not bool(valid.any()):
        raise ValueError("segmentation batch contains only ignored labels")
    invalid = valid & ((labels < 0) | (labels >= num_classes))
    if bool(invalid.any()):
        bad = labels[invalid].unique().detach().cpu().tolist()
        raise ValueError(f"segmentation labels outside [0, {num_classes - 1}]: {bad}")

    loss = F.cross_entropy(
        logits.reshape(-1, num_classes),
        labels.reshape(-1),
        weight=class_weights,
        ignore_index=ignore_index,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("segmentation cross-entropy is NaN or Inf")
    return loss


class WeightedSegmentationLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, ignore_index: int = -1):
        super().__init__()
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return weighted_segmentation_cross_entropy(
            logits,
            labels,
            self.class_weights,
            ignore_index=self.ignore_index,
        )
