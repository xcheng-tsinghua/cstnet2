from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


N_PRIMITIVES = 5
CONSTRAINT_DIM = 15
INVALID_DIRECTION = (0.0, 0.0, -1.0)


def canonicalize_directions(direction: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Map opposite primitive directions to a stable representative."""
    direction = F.normalize(direction, dim=-1, eps=eps)
    x, y, z = direction.unbind(dim=-1)
    flip = (
        (z < -eps)
        | ((z.abs() <= eps) & (y < -eps))
        | ((z.abs() <= eps) & (y.abs() <= eps) & (x < -eps))
    )
    return torch.where(flip.unsqueeze(-1), -direction, direction)


def estimate_normals_pca(xyz: torch.Tensor, k: int = 16, eps: float = 1e-6) -> torch.Tensor:
    """
    Estimate per-point normals from local PCA.

    xyz: [B, N, 3]
    return: [B, N, 3]
    """
    bsz, n_points, _ = xyz.shape
    if n_points <= 2:
        normal = xyz.new_tensor(INVALID_DIRECTION).view(1, 1, 3).repeat(bsz, n_points, 1)
        return normal

    k = max(2, min(k, n_points - 1))
    dist = torch.cdist(xyz, xyz)
    knn_idx = dist.topk(k=k + 1, dim=-1, largest=False).indices[:, :, 1:]
    batch_idx = torch.arange(bsz, device=xyz.device).view(bsz, 1, 1)
    neighbors = xyz[batch_idx, knn_idx]
    centered = neighbors - neighbors.mean(dim=2, keepdim=True)
    cov = centered.transpose(-1, -2) @ centered / float(k)
    _, eigvec = torch.linalg.eigh(cov)
    normals = eigvec[..., 0]
    normals = F.normalize(normals, dim=-1, eps=eps)
    return canonicalize_directions(normals, eps=eps)


def cluster_embeddings_radius(embedding: torch.Tensor, bandwidth: float = 0.35) -> torch.Tensor:
    """
    Cluster one point cloud's embedding features by connected components.

    This is intentionally dependency-free so Stage 1 inference does not require
    sklearn or Open3D. The embedding is expected to be L2-normalized.
    """
    emb = embedding.detach().float().cpu()
    n_points = emb.shape[0]
    if n_points == 0:
        return torch.empty(0, dtype=torch.long)

    dist = torch.cdist(emb, emb)
    adjacent = dist <= bandwidth
    visited = torch.zeros(n_points, dtype=torch.bool)
    labels = torch.full((n_points,), -1, dtype=torch.long)

    label = 0
    for seed in range(n_points):
        if visited[seed]:
            continue
        queue = [seed]
        visited[seed] = True
        labels[seed] = label
        while queue:
            cur = queue.pop()
            neighbors = torch.nonzero(adjacent[cur] & ~visited, as_tuple=False).flatten().tolist()
            for nbr in neighbors:
                visited[nbr] = True
                labels[nbr] = label
                queue.append(nbr)
        label += 1

    return labels


def _unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(v, dim=-1, eps=eps)


def _pca(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = points.mean(dim=0)
    if points.shape[0] < 3:
        return center, torch.eye(3, device=points.device, dtype=points.dtype)
    centered = points - center
    cov = centered.transpose(0, 1) @ centered / float(points.shape[0])
    _, eigvec = torch.linalg.eigh(cov)
    return center, eigvec


def _fit_plane(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center, eigvec = _pca(points)
    normal = canonicalize_directions(eigvec[:, 0].view(1, 3)).view(3)
    foot = normal * torch.dot(normal, center)
    dim = points.new_tensor(-1.0)
    return normal, dim, foot


def _fit_cylinder(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center, eigvec = _pca(points)
    axis = canonicalize_directions(eigvec[:, -1].view(1, 3)).view(3)
    foot = center - axis * torch.dot(axis, center)
    radial = torch.cross(points - foot, axis.expand_as(points), dim=1).norm(dim=1)
    radius = radial.mean() if radial.numel() > 0 else points.new_tensor(0.0)
    return axis, radius, foot


def _fit_cone(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center, eigvec = _pca(points)
    axis = canonicalize_directions(eigvec[:, -1].view(1, 3)).view(3)
    proj = (points - center) @ axis
    apex = center + proj.min() * axis
    v = points - apex
    axial = (v @ axis).abs().clamp_min(1e-6)
    radial = (v - (v @ axis).unsqueeze(1) * axis).norm(dim=1)
    semi_angle = torch.atan2(radial.mean(), axial.mean())
    return axis, semi_angle, apex


def _fit_sphere(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = points.mean(dim=0)
    radius = (points - center).norm(dim=1).mean()
    direction = points.new_tensor(INVALID_DIRECTION)
    return direction, radius, center


def _fit_other(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return points.new_tensor(INVALID_DIRECTION), points.new_tensor(-1.0), points.new_zeros(3)


def _majority(values: torch.Tensor, n_classes: int = N_PRIMITIVES) -> int:
    if values.numel() == 0:
        return n_classes - 1
    return int(torch.bincount(values.long(), minlength=n_classes).argmax().item())


def assemble_constraints_from_stage1(
    xyz: torch.Tensor,
    cluster_embedding: torch.Tensor,
    log_primitive: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
    cluster_bandwidth: float = 0.35,
    normal_k: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Convert Stage 1 primitive logits and clustering embeddings to constraints.

    xyz: [B, N, 3]
    cluster_embedding: [B, N, D]
    log_primitive: [B, N, 5]
    normals: optional [B, N, 3]
    """
    bsz, n_points, _ = xyz.shape
    device = xyz.device
    dtype = xyz.dtype

    pmt_idx = log_primitive.argmax(dim=-1)
    primitive_type = torch.zeros(bsz, n_points, N_PRIMITIVES, device=device, dtype=dtype)
    direction = torch.zeros(bsz, n_points, 3, device=device, dtype=dtype)
    dimension = torch.full((bsz, n_points), -1.0, device=device, dtype=dtype)
    location = torch.zeros(bsz, n_points, 3, device=device, dtype=dtype)

    if normals is None:
        continuity = estimate_normals_pca(xyz, k=normal_k).to(dtype=dtype)
    else:
        continuity = F.normalize(normals.to(device=device, dtype=dtype), dim=-1, eps=1e-6)

    for b in range(bsz):
        labels = cluster_embeddings_radius(cluster_embedding[b], bandwidth=cluster_bandwidth).to(device)
        for cluster_id in labels.unique(sorted=True):
            mask = labels == cluster_id
            points = xyz[b, mask]
            prim = _majority(pmt_idx[b, mask])
            primitive_type[b, mask, prim] = 1.0

            if prim == 0:
                fit_dir, fit_dim, fit_loc = _fit_plane(points)
            elif prim == 1:
                fit_dir, fit_dim, fit_loc = _fit_cylinder(points)
            elif prim == 2:
                fit_dir, fit_dim, fit_loc = _fit_cone(points)
            elif prim == 3:
                fit_dir, fit_dim, fit_loc = _fit_sphere(points)
            else:
                fit_dir, fit_dim, fit_loc = _fit_other(points)

            direction[b, mask] = fit_dir.to(dtype=dtype)
            dimension[b, mask] = fit_dim.to(dtype=dtype)
            location[b, mask] = fit_loc.to(dtype=dtype)

    return {
        "primitive_type": primitive_type,
        "direction": direction,
        "dimension": dimension,
        "continuity": continuity,
        "location": location,
    }


def constraints_to_tensor(constraints: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            constraints["primitive_type"],
            constraints["direction"],
            constraints["dimension"].unsqueeze(-1),
            constraints["continuity"],
            constraints["location"],
        ],
        dim=-1,
    )


def ground_truth_constraints_to_tensor(
    pmt: torch.Tensor,
    direction: torch.Tensor,
    dimension: torch.Tensor,
    continuity: torch.Tensor,
    location: torch.Tensor,
    n_primitives: int = N_PRIMITIVES,
) -> torch.Tensor:
    pmt_one_hot = F.one_hot(pmt.long(), n_primitives).to(dtype=direction.dtype, device=direction.device)
    return torch.cat(
        [
            pmt_one_hot,
            direction,
            dimension.unsqueeze(-1) if dimension.dim() == pmt.dim() else dimension,
            continuity,
            location,
        ],
        dim=-1,
    )


def split_constraint_tensor(constraints: torch.Tensor) -> Dict[str, torch.Tensor]:
    if constraints.shape[-1] != CONSTRAINT_DIM:
        raise ValueError(f"expected constraint dim {CONSTRAINT_DIM}, got {constraints.shape[-1]}")
    return {
        "primitive_type": constraints[..., 0:5],
        "direction": constraints[..., 5:8],
        "dimension": constraints[..., 8:9],
        "continuity": constraints[..., 9:12],
        "location": constraints[..., 12:15],
    }
