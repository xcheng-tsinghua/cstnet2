from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def _metrics_from_confusion(confusion: torch.Tensor, prefix: str) -> dict[str, Any]:
    confusion = confusion.to(dtype=torch.float64)
    tp = confusion.diag()
    gt = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    fp = predicted - tp
    fn = gt - tp
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    iou = tp / (tp + fp + fn).clamp_min(1.0)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)
    dice = f1.clone()
    valid_classes = gt > 0
    valid_count = valid_classes.sum().clamp_min(1)
    mean_accuracy = recall[valid_classes].sum() / valid_count
    mean_iou = iou[valid_classes].sum() / valid_count
    mean_f1 = f1[valid_classes].sum() / valid_count
    mean_dice = dice[valid_classes].sum() / valid_count
    overall_accuracy = tp.sum() / confusion.sum().clamp_min(1.0)
    return {
        f"{prefix}_overall_accuracy": float(overall_accuracy.item()),
        f"{prefix}_mean_class_accuracy": float(mean_accuracy.item()),
        f"{prefix}_per_class_precision": precision.cpu().tolist(),
        f"{prefix}_per_class_recall": recall.cpu().tolist(),
        f"{prefix}_per_class_iou": iou.cpu().tolist(),
        f"{prefix}_per_class_f1": f1.cpu().tolist(),
        f"{prefix}_mean_f1": float(mean_f1.item()),
        f"{prefix}_per_class_dice": dice.cpu().tolist(),
        f"{prefix}_mean_dice": float(mean_dice.item()),
        f"{prefix}_mean_iou": float(mean_iou.item()),
        f"{prefix}_confusion_matrix": confusion.to(dtype=torch.int64).cpu().tolist(),
    }


class SegmentationMetrics:
    """Accumulate exact point and mean-logit Face confusion matrices."""

    def __init__(self, num_classes: int, ignore_index: int = -1):
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.point_confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        self.face_confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        self._synchronized = False

    def _accumulate(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        face_level: bool,
    ) -> None:
        target = target.detach().long().reshape(-1).cpu()
        prediction = prediction.detach().long().reshape(-1).cpu()
        valid = target != self.ignore_index
        target = target[valid]
        prediction = prediction[valid]
        if target.numel() == 0:
            return
        if bool(((target < 0) | (target >= self.num_classes)).any()):
            raise ValueError("metric target contains an out-of-range class")
        if bool(((prediction < 0) | (prediction >= self.num_classes)).any()):
            raise ValueError("metric prediction contains an out-of-range class")
        flat = target * self.num_classes + prediction
        update = torch.bincount(flat, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )
        if face_level:
            self.face_confusion += update
        else:
            self.point_confusion += update

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        face_ids: torch.Tensor,
    ) -> None:
        if logits.ndim != 3 or logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"expected logits [B, N, {self.num_classes}], got {tuple(logits.shape)}"
            )
        if labels.shape != logits.shape[:2] or face_ids.shape != logits.shape[:2]:
            raise ValueError("labels and face_ids must align with point logits")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("cannot evaluate non-finite segmentation logits")

        self._accumulate(labels, logits.argmax(dim=-1), face_level=False)
        for batch_id in range(logits.shape[0]):
            for face_id in face_ids[batch_id].unique(sorted=True):
                if int(face_id) < 0:
                    continue
                face_mask = face_ids[batch_id] == face_id
                face_labels = labels[batch_id, face_mask]
                face_labels = face_labels[face_labels != self.ignore_index]
                if face_labels.numel() == 0:
                    continue
                unique_labels = face_labels.unique()
                if unique_labels.numel() != 1:
                    raise ValueError(
                        f"Face {int(face_id)} contains inconsistent ground-truth labels: "
                        f"{unique_labels.detach().cpu().tolist()}"
                    )
                face_prediction = logits[batch_id, face_mask].mean(dim=0).argmax()
                self._accumulate(unique_labels, face_prediction.view(1), face_level=True)

    def synchronize(self) -> None:
        if self._synchronized or not (dist.is_available() and dist.is_initialized()):
            return
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        point = self.point_confusion.to(device)
        face = self.face_confusion.to(device)
        dist.all_reduce(point, op=dist.ReduceOp.SUM)
        dist.all_reduce(face, op=dist.ReduceOp.SUM)
        self.point_confusion = point.cpu()
        self.face_confusion = face.cpu()
        self._synchronized = True

    def compute(self) -> dict[str, Any]:
        self.synchronize()
        point = _metrics_from_confusion(self.point_confusion, "point")
        face = _metrics_from_confusion(self.face_confusion, "face")
        return {**point, **face}
