from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from functional.constraints import cluster_embeddings_radius


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

    point_acc = (pred == target).float().mean() if target.numel() > 0 else torch.zeros((), device=device)
    per_class_acc = []
    per_class_f1 = []
    per_class_iou = []

    for cls_idx in range(n_classes):
        pred_c = pred == cls_idx
        target_c = target == cls_idx
        tp = (pred_c & target_c).sum().float()
        fp = (pred_c & ~target_c).sum().float()
        fn = (~pred_c & target_c).sum().float()
        support = target_c.sum().float()

        acc = tp / support.clamp_min(1.0)
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        iou = tp / (tp + fp + fn).clamp_min(1.0)

        # Missing classes should not make macro metrics look artificially bad.
        if support <= 0:
            acc = torch.zeros((), device=device)
            f1 = torch.zeros((), device=device)
            iou = torch.zeros((), device=device)

        per_class_acc.append(acc)
        per_class_f1.append(f1)
        per_class_iou.append(iou)

    per_class_acc_t = torch.stack(per_class_acc)
    per_class_f1_t = torch.stack(per_class_f1)
    per_class_iou_t = torch.stack(per_class_iou)
    valid_classes = torch.stack([(target == i).any() for i in range(n_classes)]).to(device=device)
    valid_float = valid_classes.float()
    valid_denom = valid_float.sum().clamp_min(1.0)

    macro_f1 = (per_class_f1_t * valid_float).sum() / valid_denom
    mean_iou = (per_class_iou_t * valid_float).sum() / valid_denom

    return {
        "pmt_acc": point_acc,
        "pmt_per_class_acc": per_class_acc_t,
        "pmt_macro_f1": macro_f1,
        "pmt_per_class_iou": per_class_iou_t,
        "pmt_miou": mean_iou,
    }
