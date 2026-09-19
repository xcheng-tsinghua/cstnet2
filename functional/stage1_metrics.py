from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn.functional as F

from functional.constraints import (
    CLUSTER_METHODS,
    cluster_embeddings_mean_shift,
    cluster_embeddings_radius,
)


CONSTRAINT_ATTRIBUTE_METRIC_SPECS = {
    "direction_mean_angular_error_deg": (
        "_constraint_attribute_sum/direction_angular_error_deg",
        "_constraint_attribute_count/direction",
        "direction_valid_points",
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

ATTRIBUTE_TRIM_RATIOS = {"trim1p": 0.01, "trim5p": 0.05, "trim10p": 0.10}
TRIMMED_ATTRIBUTE_METRIC_SPECS = {
    f"{section}/{name}": (
        sum_key.replace("/", f"/{section}/", 1),
        count_key.replace("/", f"/{section}/", 1),
        f"{section}/{valid_count_name}",
    )
    for section in ATTRIBUTE_TRIM_RATIOS
    for name, (sum_key, count_key, valid_count_name)
    in CONSTRAINT_ATTRIBUTE_METRIC_SPECS.items()
}
TRIMMED_ATTRIBUTE_ACCUMULATOR_KEYS = frozenset(
    key for sum_key, count_key, _ in TRIMMED_ATTRIBUTE_METRIC_SPECS.values()
    for key in (sum_key, count_key)
)

INRANGE_ATTRIBUTE_METRIC_SPECS = {
    f"inrange/{name}": (
        sum_key.replace("/", "/inrange/", 1),
        count_key.replace("/", "/inrange/", 1),
        f"inrange/{valid_count_name}",
    )
    for name, (sum_key, count_key, valid_count_name)
    in CONSTRAINT_ATTRIBUTE_METRIC_SPECS.items()
}
INRANGE_ATTRIBUTE_ACCUMULATOR_KEYS = frozenset(
    key for sum_key, count_key, _ in INRANGE_ATTRIBUTE_METRIC_SPECS.values()
    for key in (sum_key, count_key)
)
ATTRIBUTE_METRIC_SECTIONS = frozenset((*ATTRIBUTE_TRIM_RATIOS, "inrange"))


def _primitive_mask(pmt_gt: torch.Tensor, valid_types: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(pmt_gt, dtype=torch.bool)
    for primitive_type in valid_types:
        mask |= pmt_gt == primitive_type
    return mask


def _angular_error_sum_and_count(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    trim_ratio: float = 0.0,
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
    pred_unit = F.normalize(torch.where(valid[..., None], prediction.float(), 0.0), dim=-1, eps=eps)
    target_unit = F.normalize(torch.where(valid[..., None], target.float(), 0.0), dim=-1, eps=eps)
    # Plane normals and cylinder/cone axes are unoriented. The absolute dot
    # product makes nearly antiparallel vectors robustly equivalent even near
    # the discontinuous dir_unify sign boundary.
    cosine = (pred_unit * target_unit).sum(dim=-1).abs().clamp(0.0, 1.0)
    error_deg = torch.where(valid, torch.acos(cosine) * (180.0 / math.pi), 0.0)
    return _trimmed_error_sum_and_count(error_deg, valid, trim_ratio)


def _trimmed_error_sum_and_count(
    errors: torch.Tensor,
    valid_mask: torch.Tensor,
    trim_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop each cloud's largest errors and return additive accumulators."""
    if errors.shape != valid_mask.shape or errors.ndim != 2:
        raise ValueError("attribute errors and masks must have shape [B, N]")
    if not 0.0 <= trim_ratio < 1.0:
        raise ValueError("trim_ratio must be in [0, 1)")

    if trim_ratio == 0.0:
        return (
            torch.where(valid_mask, errors, 0.0).sum(),
            valid_mask.sum().to(dtype=torch.float32),
        )

    counts = valid_mask.sum(dim=1)
    retained = counts - (counts.double() * trim_ratio).floor().long()
    ordered = errors.float().masked_fill(~valid_mask, float("inf")).sort(dim=1).values
    keep = torch.arange(errors.shape[1], device=errors.device)[None, :] < retained[:, None]
    # Sum retained errors directly: subtracting huge outliers can erase small errors.
    return torch.where(keep, ordered, 0.0).sum(), retained.sum().float()


@torch.no_grad()
def evaluate_constraint_attribute_metrics(
    mad_pred: torch.Tensor,
    dim_pred: torch.Tensor,
    loc_pred: torch.Tensor,
    pmt_gt: torch.Tensor,
    mad_gt: torch.Tensor,
    dim_gt: torch.Tensor,
    loc_gt: torch.Tensor,
    trim_ratio: float = 0.0,
    include_trimmed: bool = False,
    range_masks: Mapping[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    """Return additive accumulators for exact epoch-level constraint errors.

    When ``trim_ratio`` is nonzero, the largest errors are removed separately
    from each point cloud before the retained sums and counts are aggregated.
    ``range_masks`` additionally emits inrange metrics using the dataset's
    primitive-instance range masks. Original and trimmed metrics are unaffected.
    """
    direction_sum, direction_count = _angular_error_sum_and_count(
        mad_pred,
        mad_gt,
        _primitive_mask(pmt_gt, (0, 1, 2)),
        trim_ratio=trim_ratio,
    )
    dimension_mask = (
        _primitive_mask(pmt_gt, (1, 2, 3))
        & torch.isfinite(dim_pred)
        & torch.isfinite(dim_gt)
    )
    dimension_errors = (dim_pred.float() - dim_gt.float()).abs()
    dimension_sum, dimension_count = _trimmed_error_sum_and_count(
        dimension_errors,
        dimension_mask,
        trim_ratio,
    )

    location_mask = (
        _primitive_mask(pmt_gt, (0, 1, 2, 3))
        & torch.isfinite(loc_pred).all(dim=-1)
        & torch.isfinite(loc_gt).all(dim=-1)
    )
    location_errors = (loc_pred.float() - loc_gt.float()).norm(dim=-1)
    location_sum, location_count = _trimmed_error_sum_and_count(
        location_errors,
        location_mask,
        trim_ratio,
    )

    output = {
        "_constraint_attribute_sum/direction_angular_error_deg": direction_sum,
        "_constraint_attribute_count/direction": direction_count,
        "_constraint_attribute_sum/dimension_absolute_error": dimension_sum,
        "_constraint_attribute_count/dimension": dimension_count,
        "_constraint_attribute_sum/location_distance_error": location_sum,
        "_constraint_attribute_count/location": location_count,
    }
    if include_trimmed:
        for section, ratio in ATTRIBUTE_TRIM_RATIOS.items():
            trimmed = evaluate_constraint_attribute_metrics(
                mad_pred, dim_pred, loc_pred, pmt_gt, mad_gt, dim_gt, loc_gt,
                trim_ratio=ratio,
            )
            output.update({
                key.replace("/", f"/{section}/", 1): value
                for key, value in trimmed.items()
            })
    if range_masks is not None:
        for mask in range_masks.values():
            if mask.shape != pmt_gt.shape or mask.dtype != torch.bool:
                raise ValueError("range masks must be boolean with shape [B, N]")
        dim_inrange = dimension_mask & range_masks.get("dim_valid_mask", torch.ones_like(dimension_mask))
        loc_inrange = location_mask & range_masks.get("loc_valid_mask", torch.ones_like(location_mask))
        dim_sum, dim_count = _trimmed_error_sum_and_count(dimension_errors, dim_inrange, 0.0)
        loc_sum, loc_count = _trimmed_error_sum_and_count(location_errors, loc_inrange, 0.0)
        dir_inrange = _primitive_mask(pmt_gt, (0, 1, 2))
        if "mad_valid_mask" in range_masks:
            dir_inrange = dir_inrange & range_masks["mad_valid_mask"]
        if "mad_valid_mask" not in range_masks and trim_ratio == 0:
            dir_sum, dir_count = direction_sum, direction_count
        else:
            dir_sum, dir_count = _angular_error_sum_and_count(mad_pred, mad_gt, dir_inrange)
        output.update({
            "_constraint_attribute_sum/inrange/direction_angular_error_deg": dir_sum,
            "_constraint_attribute_count/inrange/direction": dir_count,
            "_constraint_attribute_sum/inrange/dimension_absolute_error": dim_sum,
            "_constraint_attribute_count/inrange/dimension": dim_count,
            "_constraint_attribute_sum/inrange/location_distance_error": loc_sum,
            "_constraint_attribute_count/inrange/location": loc_count,
        })
    return output


def aggregate_constraint_attribute_metrics(
    metric_batches: Iterable[Mapping[str, Any]],
) -> Dict[str, float]:
    """Convert per-batch additive accumulators into exact epoch means."""
    batches = list(metric_batches)
    output: Dict[str, float] = {}
    for metric_name, (sum_key, count_key, valid_count_name) in (
        (CONSTRAINT_ATTRIBUTE_METRIC_SPECS | TRIMMED_ATTRIBUTE_METRIC_SPECS
         | INRANGE_ATTRIBUTE_METRIC_SPECS).items()
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
    *,
    predicted_affiliate_idx: torch.Tensor | None = None,
    cluster_method: str = "radius",
    mean_shift_quantile: float = 0.015,
    mean_shift_iterations: int = 20,
    mean_shift_max_clusters: int = 128,
    mean_shift_bandwidth: float | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate real inference-time clustering from predicted embeddings.

    The prediction path mirrors the selected Stage 1 inference clusterer. GT
    centers and the number of GT instances are never used to make predictions.
    """
    if cluster_method not in CLUSTER_METHODS:
        raise ValueError(
            f"unsupported cluster_method={cluster_method!r}; expected one of {CLUSTER_METHODS}"
        )
    affiliate_idx = affiliate_idx.detach().long()
    point_emb = point_emb.detach().float()
    if (
        predicted_affiliate_idx is not None
        and tuple(predicted_affiliate_idx.shape) != tuple(affiliate_idx.shape)
    ):
        raise ValueError(
            "predicted_affiliate_idx must have the same shape as affiliate_idx"
        )
    device = point_emb.device
    bsz = point_emb.shape[0]

    aris, nmis, pred_counts, gt_counts = [], [], [], []
    for b in range(bsz):
        gt = affiliate_idx[b]
        _, gt_idx = torch.unique(gt, sorted=True, return_inverse=True)
        gt_count = int(gt_idx.max().item()) + 1 if gt_idx.numel() > 0 else 0

        if predicted_affiliate_idx is not None:
            pred_idx = predicted_affiliate_idx[b].detach().long().to(device=device)
        else:
            emb = F.normalize(point_emb[b], dim=-1, eps=1e-6)
            if cluster_method == "meanshift":
                pred_idx = cluster_embeddings_mean_shift(
                    emb,
                    quantile=mean_shift_quantile,
                    iterations=mean_shift_iterations,
                    max_clusters=mean_shift_max_clusters,
                    bandwidth=mean_shift_bandwidth,
                ).to(device=device)
            else:
                pred_idx = cluster_embeddings_radius(
                    emb, bandwidth=bandwidth
                ).to(device=device)
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
