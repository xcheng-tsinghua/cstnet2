from __future__ import annotations

import torch


def _symmetric_eigh(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigh"):
        return torch.linalg.eigh(mat)
    return torch.symeig(mat, eigenvectors=True)


def _knn_local_pca(xyz: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return local PCA eigenvalues and neighbor distances.

    xyz: [B, N, 3]
    eigenvalues: [B, N, 3], ascending
    knn_dist: [B, N, K]
    """
    bsz, n_points, _ = xyz.shape
    if n_points <= 2:
        eigvals = xyz.new_zeros(bsz, n_points, 3)
        knn_dist = xyz.new_zeros(bsz, n_points, 1)
        return eigvals, knn_dist

    k = max(2, min(int(k), n_points - 1))
    dist = torch.cdist(xyz, xyz)
    knn = dist.topk(k=k + 1, dim=-1, largest=False)
    knn_idx = knn.indices[:, :, 1:]
    knn_dist = knn.values[:, :, 1:]

    batch_idx = torch.arange(bsz, device=xyz.device).view(bsz, 1, 1)
    neighbors = xyz[batch_idx, knn_idx]
    centered = neighbors - neighbors.mean(dim=2, keepdim=True)
    cov = centered.transpose(-1, -2) @ centered / float(k)
    eigvals, _ = _symmetric_eigh(cov)
    eigvals = eigvals.clamp_min(0.0)
    return eigvals, knn_dist


def build_stage1_input_features(
    xyz: torch.Tensor,
    use_curvature: bool = True,
    use_density: bool = True,
    k: int = 16,
) -> torch.Tensor:
    """
    Build normal-free local geometric features from point coordinates.

    Returned feature layout:
    - curvature lambda_min / lambda_sum [1], when use_curvature=True
    - local density mean KNN distance [1], when use_density=True
    """
    xyz = xyz.float()
    features = []
    need_pca = use_curvature or use_density
    eigvals = knn_dist = None

    if need_pca:
        eigvals, knn_dist = _knn_local_pca(xyz, k=k)

    if use_curvature:
        eig_sum = eigvals.sum(dim=-1, keepdim=True)
        curvature = eigvals[..., :1] / (eig_sum + 1e-6)
        features.append(curvature)

    if use_density:
        features.append(knn_dist.mean(dim=-1, keepdim=True))

    if len(features) == 0:
        bsz, n_points, _ = xyz.shape
        return xyz.new_empty(bsz, n_points, 0)
    return torch.cat(features, dim=-1)


def stage1_feature_dim(use_extra_features: bool) -> int:
    if not use_extra_features:
        return 0
    return 2
