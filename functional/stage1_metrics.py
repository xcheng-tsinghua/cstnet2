from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn.functional as F

from functional.constraints import cluster_embeddings_radius


CONSTRAINT_ATTRIBUTE_METRIC_SPECS = {
    "direction_mean_angular_error_deg": (
        "_constraint_attribute_sum/direction_angular_error_deg",
        "_constraint_attribute_count/direction",
        "direction_valid_points",
    ),
    "continuity_mean_angular_error_deg": (
        "_constraint_attribute_sum/continuity_angular_error_deg",
        "_constraint_attribute_count/continuity",
        "continuity_valid_points",
    ),
    "dimension_mean_absolute_error": (
        "_constraint_attribute_sum/dimension_absolute_error",
        "_constraint_attribute_count/dimension",
        "dimension_valid_points",
    ),
    "location_mean_distance_error": (
        "_constraint_attribute_sum/location_distance_error",
        "_constraint_attribute_count/location",
        "location_valid_points",
    ),
}

CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS = frozenset(
    key
    for _, (sum_key, count_key, _) in CONSTRAINT_ATTRIBUTE_METRIC_SPECS.items()
    for key in (sum_key, count_key)
)


def _primitive_mask(pmt_gt: torch.Tensor, valid_types: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(pmt_gt, dtype=torch.bool)
    for primitive_type in valid_types:
        mask |= pmt_gt == primitive_type
    return mask


def _canonicalize_direction(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply the project's dir_unify sign convention to normalized vectors."""
    x, y, z = vectors.unbind(dim=-1)
    z_zero = z.abs() <= eps
    y_zero = y.abs() <= eps
    flip = (z < -eps) | (z_zero & (y < -eps)) | (
        z_zero & y_zero & (x < -eps)
    )
    return torch.where(flip.unsqueeze(-1), -vectors, vectors)


def _angular_error_sum_and_count(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    canonicalize: bool,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_norm = prediction.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    valid = (
        mask
        & torch.isfinite(prediction).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & (pred_norm > eps)
        & (target_norm > eps)
    )
    count = valid.sum().to(dtype=torch.float32)
    if not bool(valid.any()):
        return prediction.new_zeros((), dtype=torch.float32), count

    pred_unit = F.normalize(prediction[valid].float(), dim=-1, eps=eps)
    target_unit = F.normalize(target[valid].float(), dim=-1, eps=eps)
    if canonicalize:
        pred_unit = _canonicalize_direction(pred_unit, eps=eps)
        target_unit = _canonicalize_direction(target_unit, eps=eps)
    cosine = (pred_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    error_deg = torch.acos(cosine) * (180.0 / math.pi)
    return error_deg.sum(), count


@torch.no_grad()
def evaluate_constraint_attribute_metrics(
    mad_pred: torch.Tensor,
    dim_pred: torch.Tensor,
    nor_pred: torch.Tensor,
    loc_pred: torch.Tensor,
    pmt_gt: torch.Tensor,
    mad_gt: torch.Tensor,
    dim_gt: torch.Tensor,
    nor_gt: torch.Tensor,
    loc_gt: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return additive accumulators for exact epoch-level constraint errors."""
    direction_sum, direction_count = _angular_error_sum_and_count(
        mad_pred,
        mad_gt,
        _primitive_mask(pmt_gt, (0, 1, 2)),
        canonicalize=True,
    )
    continuity_sum, continuity_count = _angular_error_sum_and_count(
        nor_pred,
        nor_gt,
        _primitive_mask(pmt_gt, (0, 1, 2, 3, 4)),
        canonicalize=False,
    )

    dimension_mask = (
        _primitive_mask(pmt_gt, (1, 2, 3))
        & torch.isfinite(dim_pred)
        & torch.isfinite(dim_gt)
    )
    dimension_errors = (dim_pred.float() - dim_gt.float()).abs()
    dimension_sum = (
        dimension_errors[dimension_mask].sum()
        if bool(dimension_mask.any())
        else dimension_errors.new_zeros(())
    )
    dimension_count = dimension_mask.sum().to(dtype=torch.float32)

    location_mask = (
        _primitive_mask(pmt_gt, (0, 1, 2, 3))
        & torch.isfinite(loc_pred).all(dim=-1)
        & torch.isfinite(loc_gt).all(dim=-1)
    )
    location_errors = (loc_pred.float() - loc_gt.float()).norm(dim=-1)
    location_sum = (
        location_errors[location_mask].sum()
        if bool(location_mask.any())
        else location_errors.new_zeros(())
    )
    location_count = location_mask.sum().to(dtype=torch.float32)

    return {
        "_constraint_attribute_sum/direction_angular_error_deg": direction_sum,
        "_constraint_attribute_count/direction": direction_count,
        "_constraint_attribute_sum/continuity_angular_error_deg": continuity_sum,
        "_constraint_attribute_count/continuity": continuity_count,
        "_constraint_attribute_sum/dimension_absolute_error": dimension_sum,
        "_constraint_attribute_count/dimension": dimension_count,
        "_constraint_attribute_sum/location_distance_error": location_sum,
        "_constraint_attribute_count/location": location_count,
    }


def aggregate_constraint_attribute_metrics(
    metric_batches: Iterable[Mapping[str, Any]],
) -> Dict[str, float]:
    """Convert per-batch additive accumulators into exact epoch means."""
    batches = list(metric_batches)
    output: Dict[str, float] = {}
    for metric_name, (sum_key, count_key, valid_count_name) in (
        CONSTRAINT_ATTRIBUTE_METRIC_SPECS.items()
    ):
        available = [
            batch for batch in batches if sum_key in batch and count_key in batch
        ]
        if not available:
            continue
        total_sum = sum(float(torch.as_tensor(batch[sum_key]).item()) for batch in available)
        total_count = sum(
            float(torch.as_tensor(batch[count_key]).item()) for batch in available
        )
        output[metric_name] = total_sum / max(total_count, 1.0)
        output[valid_count_name] = total_count
    return output


def _contingency_matrix(y_true_idx: torch.Tensor, y_pred_idx: torch.Tensor) -> torch.Tensor:
    n_true = int(y_true_idx.max().item()) + 1 if y_true_idx.numel() > 0 else 0
    n_pred = int(y_pred_idx.max().item()) + 1 if y_pred_idx.numel() > 0 else 0
    mat = torch.zeros((n_true, n_pred), device=y_true_idx.device, dtype=torch.float32)
    if y_true_idx.numel() == 0:
        return mat
    ones = torch.ones_like(y_true_idx, dtype=torch.float32)
    mat.index_put_((y_true_idx, y_pred_idx), ones, accumulate=True)
    return mat


def _ari_from_contingency(cont: torch.Tensor) -> torch.Tensor:
    n = cont.sum()
    if n <= 1:
        return torch.zeros((), device=cont.device, dtype=torch.float32)

    row_sum = cont.sum(dim=1)
    col_sum = cont.sum(dim=0)
    comb_cont = (cont * (cont - 1.0) * 0.5).sum()
    comb_row = (row_sum * (row_sum - 1.0) * 0.5).sum()
    comb_col = (col_sum * (col_sum - 1.0) * 0.5).sum()
    comb_n = n * (n - 1.0) * 0.5
    expected = comb_row * comb_col / (comb_n + 1e-12)
    max_index = 0.5 * (comb_row + comb_col)
    return (comb_cont - expected) / (max_index - expected + 1e-12)


def _nmi_from_contingency(cont: torch.Tensor) -> torch.Tensor:
    n = cont.sum()
    if n <= 0:
        return torch.zeros((), device=cont.device, dtype=torch.float32)

    p_ij = cont / n
    p_i = p_ij.sum(dim=1, keepdim=True)
    p_j = p_ij.sum(dim=0, keepdim=True)
    expected = p_i @ p_j
    valid = p_ij > 0
    mi = (p_ij[valid] * torch.log((p_ij[valid] + 1e-12) / (expected[valid] + 1e-12))).sum()
    h_i = -(p_i[p_i > 0] * torch.log(p_i[p_i > 0] + 1e-12)).sum()
    h_j = -(p_j[p_j > 0] * torch.log(p_j[p_j > 0] + 1e-12)).sum()
    return (2.0 * mi) / (h_i + h_j + 1e-12)


def evaluate_predicted_clustering(
    affiliate_idx: torch.Tensor,
    point_emb: torch.Tensor,
    bandwidth: float = 0.35,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate real inference-time clustering from predicted embeddings.

    The prediction path intentionally mirrors Stage 1 inference: normalize the
    embedding, run radius connected components, then compare predicted cluster
    ids with GT primitive-instance ids. GT centers are not used.
    """
    affiliate_idx = affiliate_idx.detach().long()
    point_emb = point_emb.detach().float()
    device = point_emb.device
    bsz = point_emb.shape[0]

    aris, nmis, pred_counts, gt_counts = [], [], [], []
    for b in range(bsz):
        gt = affiliate_idx[b]
        _, gt_idx = torch.unique(gt, sorted=True, return_inverse=True)
        gt_count = int(gt_idx.max().item()) + 1 if gt_idx.numel() > 0 else 0

        emb = F.normalize(point_emb[b], dim=-1, eps=1e-6)
        pred_idx = cluster_embeddings_radius(emb, bandwidth=bandwidth).to(device=device)
        _, pred_idx = torch.unique(pred_idx.long(), sorted=True, return_inverse=True)
        pred_count = int(pred_idx.max().item()) + 1 if pred_idx.numel() > 0 else 0

        if gt_count <= 1 and pred_count <= 1:
            ari = torch.ones((), device=device, dtype=torch.float32)
            nmi = torch.ones((), device=device, dtype=torch.float32)
        else:
            cont = _contingency_matrix(gt_idx.to(device), pred_idx)
            ari = _ari_from_contingency(cont)
            nmi = _nmi_from_contingency(cont)

        aris.append(ari)
        nmis.append(nmi)
        pred_counts.append(torch.tensor(float(pred_count), device=device))
        gt_counts.append(torch.tensor(float(gt_count), device=device))

    return {
        "cluster_ari_real": torch.stack(aris).mean() if aris else torch.zeros((), device=device),
        "cluster_nmi_real": torch.stack(nmis).mean() if nmis else torch.zeros((), device=device),
        "cluster_pred_count": torch.stack(pred_counts).mean() if pred_counts else torch.zeros((), device=device),
        "cluster_gt_count": torch.stack(gt_counts).mean() if gt_counts else torch.zeros((), device=device),
    }


def evaluate_primitive_metrics(
    log_pmt: torch.Tensor,
    pmt_gt: torch.Tensor,
    n_classes: int = 5,
) -> Dict[str, torch.Tensor]:
    """Compute point primitive classification metrics without external deps."""
    pred = log_pmt.detach().argmax(dim=-1).long().reshape(-1)
    target = pmt_gt.detach().long().reshape(-1)
    device = log_pmt.device
    if target.numel() == 0:
        confusion = torch.zeros((n_classes, n_classes), device=device, dtype=torch.float32)
    else:
        flat_indices = target * n_classes + pred
        confusion = torch.bincount(
            flat_indices, minlength=n_classes * n_classes
        ).reshape(n_classes, n_classes).float()
    return primitive_metrics_from_confusion(confusion)


def primitive_metrics_from_confusion(confusion: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Derive exact epoch metrics from a target-row/prediction-column matrix."""
    confusion = confusion.float()
    tp = confusion.diag()
    gt_hist = confusion.sum(dim=1)
    pred_hist = confusion.sum(dim=0)
    fp = pred_hist - tp
    fn = gt_hist - tp

    recall = tp / gt_hist.clamp_min(1.0)
    precision = tp / (tp + fp).clamp_min(1.0)
    per_class_f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    per_class_iou = tp / (tp + fp + fn).clamp_min(1.0)
    valid = gt_hist > 0
    valid_float = valid.float()
    valid_count = valid_float.sum().clamp_min(1.0)

    recall = torch.where(valid, recall, torch.zeros_like(recall))
    per_class_f1 = torch.where(valid, per_class_f1, torch.zeros_like(per_class_f1))
    per_class_iou = torch.where(valid, per_class_iou, torch.zeros_like(per_class_iou))
    point_acc = tp.sum() / confusion.sum().clamp_min(1.0)

    return {
        "pmt_acc": point_acc,
        "pmt_gt_histogram": gt_hist,
        "pmt_pred_histogram": pred_hist,
        "pmt_confusion_matrix": confusion,
        "pmt_per_class_acc": recall,
        "pmt_per_class_recall": recall,
        "pmt_per_class_precision": precision,
        "pmt_per_class_f1": per_class_f1,
        "pmt_macro_f1": (per_class_f1 * valid_float).sum() / valid_count,
        "pmt_per_class_iou": per_class_iou,
        "pmt_miou": (per_class_iou * valid_float).sum() / valid_count,
    }


def primitive_prediction_collapsed(
    predicted_histogram: torch.Tensor,
    threshold: float = 0.95,
) -> bool:
    """Return True when one primitive receives more than threshold of predictions."""
    histogram = torch.as_tensor(predicted_histogram).float()
    total = histogram.sum()
    return bool(total > 0 and histogram.max() / total > threshold)
