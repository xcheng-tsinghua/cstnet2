from __future__ import annotations

import torch
import torch.nn.functional as F

from functional.constraints import canonicalize_directions


def _symmetric_eigh(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigh"):
        return torch.linalg.eigh(mat)
    return torch.symeig(mat, eigenvectors=True)


def _knn_local_pca(xyz: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return local PCA eigenvalues/eigenvectors and neighbor distances.

    xyz: [B, N, 3]
    eigenvalues: [B, N, 3], ascending
    eigenvectors: [B, N, 3, 3], columns match ascending eigenvalues
    knn_dist: [B, N, K]
    """
    bsz, n_points, _ = xyz.shape
    if n_points <= 2:
        eigvals = xyz.new_zeros(bsz, n_points, 3)
        eigvecs = torch.eye(3, device=xyz.device, dtype=xyz.dtype).view(1, 1, 3, 3).repeat(bsz, n_points, 1, 1)
        knn_dist = xyz.new_zeros(bsz, n_points, 1)
        return eigvals, eigvecs, knn_dist

    k = max(2, min(int(k), n_points - 1))
    dist = torch.cdist(xyz, xyz)
    knn = dist.topk(k=k + 1, dim=-1, largest=False)
    knn_idx = knn.indices[:, :, 1:]
    knn_dist = knn.values[:, :, 1:]

    batch_idx = torch.arange(bsz, device=xyz.device).view(bsz, 1, 1)
    neighbors = xyz[batch_idx, knn_idx]
    centered = neighbors - neighbors.mean(dim=2, keepdim=True)
    cov = centered.transpose(-1, -2) @ centered / float(k)
    eigvals, eigvecs = _symmetric_eigh(cov)
    eigvals = eigvals.clamp_min(0.0)
    return eigvals, eigvecs, knn_dist


def build_stage1_input_features(
    xyz: torch.Tensor,
    normals: torch.Tensor | None = None,
    use_normals: bool = True,
    use_curvature: bool = True,
    k: int = 16,
) -> torch.Tensor:
    """
    Build optional Stage 1 point features from normals and local PCA.

    Returned feature layout:
    - normal [3], when use_normals=True
    - curvature lambda_min / lambda_sum [1], when use_curvature=True
    - local density mean KNN distance [1], when use_curvature=True
    """
    xyz = xyz.float()
    features = []
    need_pca = (use_normals and normals is None) or use_curvature
    eigvals = eigvecs = knn_dist = None

    if need_pca:
        eigvals, eigvecs, knn_dist = _knn_local_pca(xyz, k=k)

    if use_normals:
        if normals is None:
            normal_fea = eigvecs[..., 0]
            normal_fea = canonicalize_directions(normal_fea)
        else:
            normal_fea = F.normalize(normals.float(), dim=-1, eps=1e-6)
        features.append(normal_fea)

    if use_curvature:
        eig_sum = eigvals.sum(dim=-1, keepdim=True)
        curvature = eigvals[..., :1] / (eig_sum + 1e-6)
        density = knn_dist.mean(dim=-1, keepdim=True)
        features.extend([curvature, density])

    if len(features) == 0:
        bsz, n_points, _ = xyz.shape
        return xyz.new_empty(bsz, n_points, 0)
    return torch.cat(features, dim=-1)


def stage1_feature_dim(use_extra_features: bool, normal_source: str = "gt") -> int:
    if not use_extra_features:
        return 0
    normal_dim = 0 if normal_source == "none" else 3
    return normal_dim + 2
