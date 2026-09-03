from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


N_PRIMITIVES = 5
CONSTRAINT_DIM = 12  # 5 (primitive_type) + 3 (direction) + 1 (dimension) + 3 (location)
INVALID_DIRECTION = (0.0, 0.0, 0.0)
CLUSTER_METHODS = ("radius", "meanshift")


def zero_invalid_constraint_components(
    primitive_index: torch.Tensor,
    direction: torch.Tensor,
    dimension: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attributes with invalid direction and dimension set to zero."""
    if direction.shape != (*primitive_index.shape, 3):
        raise ValueError(
            "direction shape must equal primitive shape followed by 3; "
            f"got primitive={tuple(primitive_index.shape)}, "
            f"direction={tuple(direction.shape)}"
        )
    if dimension.shape not in (
        primitive_index.shape,
        (*primitive_index.shape, 1),
    ):
        raise ValueError(
            "dimension shape must equal primitive shape with an optional "
            f"trailing singleton; got primitive={tuple(primitive_index.shape)}, "
            f"dimension={tuple(dimension.shape)}"
        )

    primitive_index = primitive_index.to(device=direction.device, dtype=torch.long)
    direction_valid = (
        (primitive_index == 0)
        | (primitive_index == 1)
        | (primitive_index == 2)
    )
    dimension_valid = (
        (primitive_index == 1)
        | (primitive_index == 2)
        | (primitive_index == 3)
    )
    direction = torch.where(
        direction_valid.unsqueeze(-1), direction, torch.zeros_like(direction)
    )
    dimension_mask = (
        dimension_valid.unsqueeze(-1)
        if dimension.dim() == primitive_index.dim() + 1
        else dimension_valid
    )
    dimension = torch.where(
        dimension_mask, dimension, torch.zeros_like(dimension)
    )
    return direction, dimension


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
    _, eigvec = _symmetric_eigh(cov)
    normals = eigvec[..., 0]
    normals = F.normalize(normals, dim=-1, eps=eps)
    return canonicalize_directions(normals, eps=eps)


def cluster_embeddings_radius(embedding: torch.Tensor, bandwidth: float = 0.35) -> torch.Tensor:
    """
    Cluster one point cloud's embedding features by connected components.

    This is intentionally dependency-free so Stage 1 inference does not require
    sklearn or Open3D. The embedding is expected to be L2-normalized.
    """
    emb = F.normalize(embedding.detach().float(), dim=-1, eps=1e-6).cpu()
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


def _mean_shift_bandwidth(
    embedding: torch.Tensor,
    quantile: float,
    min_bandwidth: float,
) -> torch.Tensor:
    """Estimate a ParSeNet-style per-cloud bandwidth from embedding kNNs."""
    point_count = int(embedding.shape[0])
    if point_count <= 1:
        return embedding.new_tensor(float(min_bandwidth))
    if not 0.0 < quantile <= 1.0:
        raise ValueError("mean_shift_quantile must be in (0, 1]")
    neighbor_rank = max(1, min(point_count - 1, int(round(quantile * point_count))))
    distance_sq = (2.0 - 2.0 * (embedding @ embedding.transpose(0, 1))).clamp_min(0.0)
    # The first entry is the point itself, hence k + 1.
    kth_distance_sq = distance_sq.topk(
        k=neighbor_rank + 1, dim=1, largest=False
    ).values[:, -1]
    return kth_distance_sq.sqrt().mean().clamp_min(float(min_bandwidth))


def _mean_shift_once(
    embedding: torch.Tensor,
    bandwidth: torch.Tensor,
    iterations: int,
    convergence_tol: float,
) -> torch.Tensor:
    """Run differentiable Gaussian mean shift on the unit hypersphere."""
    shifted = embedding.clone()
    bandwidth_sq = bandwidth.square().clamp_min(1e-12)
    for _ in range(max(1, int(iterations))):
        distance_sq = (
            2.0 - 2.0 * (shifted @ embedding.transpose(0, 1))
        ).clamp_min(0.0)
        logits = -distance_sq / (2.0 * bandwidth_sq)
        # Row-wise stabilization also avoids relying on an implementation-specific
        # guarded exponential as in the original ParSeNet release.
        weights = torch.softmax(logits, dim=1)
        updated = F.normalize(weights @ embedding, dim=1, eps=1e-6)
        displacement = (updated - shifted).norm(dim=1).max()
        shifted = updated
        if float(displacement.item()) <= float(convergence_tol):
            break
    return shifted


def _mean_shift_labels(
    embedding: torch.Tensor,
    shifted: torch.Tensor,
    bandwidth: torch.Tensor,
    max_clusters: int,
) -> tuple[torch.Tensor, int]:
    """Suppress duplicate modes and assign each embedding to its nearest mode."""
    bandwidth_sq = bandwidth.square().clamp_min(1e-12)
    shifted_distance_sq = (
        2.0 - 2.0 * (shifted @ shifted.transpose(0, 1))
    ).clamp_min(0.0)
    density = torch.exp(-shifted_distance_sq / (2.0 * bandwidth_sq)).sum(dim=1)
    order = density.argsort(descending=True)
    selected: list[int] = []
    for candidate in order.tolist():
        if not selected or bool(
            (shifted_distance_sq[candidate, selected] > bandwidth_sq).all()
        ):
            selected.append(int(candidate))
    if max_clusters > 0 and len(selected) > max_clusters:
        selected = selected[:max_clusters]
    centers = shifted[torch.as_tensor(selected, device=shifted.device, dtype=torch.long)]
    labels = (embedding @ centers.transpose(0, 1)).argmax(dim=1)
    return labels, len(selected)


def cluster_embeddings_mean_shift(
    embedding: torch.Tensor,
    *,
    quantile: float = 0.015,
    iterations: int = 50,
    max_clusters: int = 128,
    bandwidth: Optional[float] = None,
    min_bandwidth: float = 0.003,
    convergence_tol: float = 1e-4,
    max_bandwidth_retries: int = 8,
) -> torch.Tensor:
    """Cluster normalized embeddings with adaptive spherical Gaussian mean shift.

    This follows ParSeNet's inference structure: all embeddings are seeds, the
    bandwidth is estimated per cloud, nearby converged modes are suppressed,
    and points receive a hard nearest-mode assignment.  Unlike the historical
    reference implementation, squared distances are consistently compared with
    squared bandwidths and the kNN rank uses the actual point count.
    """
    emb = F.normalize(embedding.detach().float(), dim=-1, eps=1e-6)
    point_count = int(emb.shape[0])
    if point_count == 0:
        return torch.empty(0, device=emb.device, dtype=torch.long)
    if point_count == 1:
        return torch.zeros(1, device=emb.device, dtype=torch.long)
    if iterations < 1:
        raise ValueError("mean_shift_iterations must be positive")
    if max_clusters == 0:
        raise ValueError("mean_shift_max_clusters must be positive or negative for unlimited")
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("mean_shift_bandwidth must be positive")

    current_quantile = float(quantile)
    retry_count = 1 if bandwidth is not None else max(1, int(max_bandwidth_retries))
    labels = torch.zeros(point_count, device=emb.device, dtype=torch.long)
    for _ in range(retry_count):
        current_bandwidth = (
            emb.new_tensor(float(bandwidth))
            if bandwidth is not None
            else _mean_shift_bandwidth(emb, current_quantile, min_bandwidth)
        )
        shifted = _mean_shift_once(
            emb,
            current_bandwidth,
            iterations=iterations,
            convergence_tol=convergence_tol,
        )
        # Do not truncate before retrying: a larger adaptive bandwidth should
        # decide whether excessive modes ought to merge.
        labels, cluster_count = _mean_shift_labels(
            emb, shifted, current_bandwidth, max_clusters=-1
        )
        if max_clusters < 0 or cluster_count <= max_clusters:
            return labels
        current_quantile = min(1.0, current_quantile * 1.2)

    # Match ParSeNet's cluster-count guard if retries could not reduce the mode
    # count enough. The retained modes are the densest ones.
    labels, _ = _mean_shift_labels(
        emb, shifted, current_bandwidth, max_clusters=max_clusters
    )
    return labels


def _unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(v, dim=-1, eps=eps)


def _symmetric_eigh(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigh"):
        return torch.linalg.eigh(mat)
    return torch.symeig(mat, eigenvectors=True)


def _least_squares(a_mat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "lstsq"):
        return torch.linalg.lstsq(a_mat, rhs).solution
    return torch.lstsq(rhs, a_mat).solution[:a_mat.shape[1]]


def _pca(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = points.mean(dim=0)
    if points.shape[0] < 3:
        return center, torch.eye(3, device=points.device, dtype=points.dtype)
    centered = points - center
    cov = centered.transpose(0, 1) @ centered / float(points.shape[0])
    _, eigvec = _symmetric_eigh(cov)
    return center, eigvec


def _robust_mean(values: torch.Tensor, trim_ratio: float = 0.1) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    if values.numel() < 4 or trim_ratio <= 0:
        return values.mean()
    sorted_values = values.sort().values
    keep = max(1, int(round(values.numel() * (1.0 - trim_ratio))))
    return sorted_values[:keep].mean()


def _orthogonal_basis(axis: torch.Tensor) -> torch.Tensor:
    """Return a stable 3x2 orthonormal basis perpendicular to ``axis``."""
    axis = _unit(axis)
    helper = torch.zeros_like(axis)
    helper[int(axis.abs().argmin().item())] = 1.0
    first = _unit(torch.cross(axis, helper, dim=0))
    second = _unit(torch.cross(axis, first, dim=0))
    return torch.stack((first, second), dim=1)


def _orient_normals_consistently(
    points: torch.Tensor,
    normals: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """Resolve local PCA normal signs by propagation over a spatial KNN graph."""
    normals = F.normalize(normals, dim=-1, eps=1e-6)
    point_count = int(points.shape[0])
    if point_count <= 1:
        return normals

    neighbor_count = max(1, min(int(k), point_count - 1))
    distances = torch.cdist(points.float(), points.float())
    neighbors = distances.topk(
        k=neighbor_count + 1, dim=-1, largest=False
    ).indices[:, 1:].detach().cpu()

    # Fitting is non-differentiable and already instance-by-instance. Doing the
    # graph traversal on CPU avoids one device synchronization per graph edge.
    work_dtype = torch.float64 if normals.dtype == torch.float64 else torch.float32
    oriented = normals.detach().to(device="cpu", dtype=work_dtype).clone()
    visited = [False] * point_count
    for seed in range(point_count):
        if visited[seed]:
            continue
        visited[seed] = True
        queue = [seed]
        while queue:
            current = queue.pop()
            for neighbor in neighbors[current].tolist():
                if visited[neighbor]:
                    continue
                if torch.dot(oriented[current], oriented[neighbor]) < 0:
                    oriented[neighbor] = -oriented[neighbor]
                visited[neighbor] = True
                queue.append(neighbor)

    return oriented.to(device=normals.device, dtype=normals.dtype)


def _fit_radial_center_2d(
    points: torch.Tensor,
    normals: torch.Tensor,
    axis: torch.Tensor,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Intersect projected normal lines to locate an axis in its normal plane."""
    basis = _orthogonal_basis(axis)
    point_2d = points @ basis
    normal_perp = normals - (normals @ axis).unsqueeze(1) * axis
    normal_norm = normal_perp.norm(dim=1)
    valid = (
        torch.isfinite(point_2d).all(dim=1)
        & torch.isfinite(normal_perp).all(dim=1)
        & (normal_norm > 1e-5)
    )
    if int(valid.sum().item()) < 2:
        return None, basis

    radial_2d = F.normalize(normal_perp[valid] @ basis, dim=1, eps=1e-6)
    point_2d = point_2d[valid]
    # A line through q with direction (dx, dy) satisfies
    # (-dy, dx) dot center = (-dy, dx) dot q.
    equations = torch.stack((-radial_2d[:, 1], radial_2d[:, 0]), dim=1)
    rhs = (equations * point_2d).sum(dim=1, keepdim=True)
    gram = equations.transpose(0, 1) @ equations
    eigenvalues, _ = _symmetric_eigh(gram)
    if (
        not torch.isfinite(eigenvalues).all()
        or float(eigenvalues[-1].item()) <= 1e-8
        or float(eigenvalues[0].item())
        <= float(eigenvalues[-1].item()) * 1e-8
    ):
        return None, basis

    try:
        center_2d = _least_squares(equations, rhs).squeeze(1)
        # Huber IRLS reduces the influence of poor local PCA normals near
        # primitive boundaries.
        for _ in range(4):
            residual = (equations @ center_2d.unsqueeze(1) - rhs).abs().squeeze(1)
            scale = residual.median().clamp_min(1e-6)
            cutoff = 2.5 * scale
            weights = torch.where(
                residual <= cutoff,
                torch.ones_like(residual),
                cutoff / residual.clamp_min(1e-6),
            )
            sqrt_weight = weights.sqrt().unsqueeze(1)
            center_2d = _least_squares(
                equations * sqrt_weight, rhs * sqrt_weight
            ).squeeze(1)
    except RuntimeError:
        return None, basis

    if not torch.isfinite(center_2d).all():
        return None, basis
    return center_2d, basis


def _fit_circle_center_2d(
    points: torch.Tensor,
    axis: torch.Tensor,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Algebraic circle fallback when usable surface normals are unavailable."""
    basis = _orthogonal_basis(axis)
    point_2d = points @ basis
    if point_2d.shape[0] < 3:
        return None, basis
    design = torch.cat(
        (
            2.0 * point_2d,
            torch.ones(
                point_2d.shape[0], 1, device=points.device, dtype=points.dtype
            ),
        ),
        dim=1,
    )
    rhs = (point_2d * point_2d).sum(dim=1, keepdim=True)
    try:
        solution = _least_squares(design, rhs).squeeze(1)
    except RuntimeError:
        return None, basis
    center_2d = solution[:2]
    if not torch.isfinite(center_2d).all():
        return None, basis
    return center_2d, basis


def _robust_linear_fit(
    x: torch.Tensor,
    y: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Fit ``y = slope * x + intercept`` with Huber IRLS."""
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.numel() < 3 or float(x.var(unbiased=False).item()) <= 1e-10:
        return None
    design = torch.stack((x, torch.ones_like(x)), dim=1)
    try:
        solution = _least_squares(design, y.unsqueeze(1)).squeeze(1)
        for _ in range(5):
            residual = (design @ solution - y).abs()
            scale = residual.median().clamp_min(1e-6)
            cutoff = 2.5 * scale
            weights = torch.where(
                residual <= cutoff,
                torch.ones_like(residual),
                cutoff / residual.clamp_min(1e-6),
            )
            sqrt_weight = weights.sqrt().unsqueeze(1)
            solution = _least_squares(
                design * sqrt_weight, y.unsqueeze(1) * sqrt_weight
            ).squeeze(1)
    except RuntimeError:
        return None
    if not torch.isfinite(solution).all():
        return None
    return solution[0], solution[1]


def _refine_cone_profile(
    point_2d: torch.Tensor,
    axial: torch.Tensor,
    center_2d: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Jointly refine the 2D axis center and linear cone radius profile."""
    radial = (point_2d - center_2d).norm(dim=1)
    linear_fit = _robust_linear_fit(axial, radial)
    if linear_fit is None:
        return None
    slope, intercept = linear_fit
    parameters = torch.cat((center_2d, slope.view(1), intercept.view(1)))
    initial_radius = (point_2d - center_2d).norm(dim=1)
    best_error = (
        initial_radius - (slope * axial + intercept)
    ).abs().median()
    best_parameters = parameters.clone()

    try:
        for _ in range(8):
            center = parameters[:2]
            slope = parameters[2]
            intercept = parameters[3]
            offset = point_2d - center
            radius = offset.norm(dim=1).clamp_min(1e-6)
            residual = radius - (slope * axial + intercept)
            jacobian = torch.cat(
                (
                    -offset / radius.unsqueeze(1),
                    -axial.unsqueeze(1),
                    -torch.ones_like(axial).unsqueeze(1),
                ),
                dim=1,
            )
            scale = residual.abs().median().clamp_min(1e-6)
            cutoff = 2.5 * scale
            weights = torch.where(
                residual.abs() <= cutoff,
                torch.ones_like(residual),
                cutoff / residual.abs().clamp_min(1e-6),
            )
            sqrt_weight = weights.sqrt().unsqueeze(1)
            update = _least_squares(
                jacobian * sqrt_weight,
                -residual.unsqueeze(1) * sqrt_weight,
            ).squeeze(1)
            if not torch.isfinite(update).all():
                break
            parameters = parameters + update
            candidate_radius = (point_2d - parameters[:2]).norm(dim=1)
            candidate_error = (
                candidate_radius - (parameters[2] * axial + parameters[3])
            ).abs().median()
            if torch.isfinite(candidate_error) and candidate_error < best_error:
                best_error = candidate_error
                best_parameters = parameters.clone()
            if float(update.norm().item()) < 1e-7:
                break
    except RuntimeError:
        pass

    if not torch.isfinite(best_parameters).all():
        return None
    return best_parameters[:2], best_parameters[2], best_parameters[3]


def _trimmed_residual_score(residual: torch.Tensor, trim_ratio: float = 0.1) -> torch.Tensor:
    values = residual.detach().abs()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return residual.new_tensor(float("inf"))
    keep = max(1, int(round(values.numel() * (1.0 - trim_ratio))))
    return values.sort().values[:keep].mean()


def _huber_irls_weights(residual: torch.Tensor) -> torch.Tensor:
    absolute = residual.detach().abs()
    scale = (1.4826 * absolute.median()).clamp_min(1e-8)
    cutoff = 2.5 * scale
    return torch.where(
        absolute <= cutoff,
        torch.ones_like(absolute),
        cutoff / absolute.clamp_min(1e-12),
    )


def _damped_irls_step(jacobian: torch.Tensor, residual: torch.Tensor) -> Optional[torch.Tensor]:
    weights = _huber_irls_weights(residual)
    sqrt_weight = weights.sqrt().unsqueeze(1)
    weighted_jacobian = jacobian * sqrt_weight
    weighted_residual = residual.unsqueeze(1) * sqrt_weight
    gram = weighted_jacobian.transpose(0, 1) @ weighted_jacobian
    rhs = -(weighted_jacobian.transpose(0, 1) @ weighted_residual).squeeze(1)
    scale = gram.diag().mean().clamp_min(1e-10)
    damping = scale * 1e-6
    gram = gram + damping * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    try:
        step = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        try:
            step = _least_squares(gram, rhs.unsqueeze(1)).squeeze(1)
        except RuntimeError:
            return None
    return step if torch.isfinite(step).all() else None


def _cylinder_residual(
    points: torch.Tensor,
    axis: torch.Tensor,
    foot: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    offset = points - foot
    axial = offset @ axis
    radial = (offset - axial.unsqueeze(1) * axis).norm(dim=1)
    return radial - radius


def _refine_cylinder_parameters(
    points: torch.Tensor,
    axis: torch.Tensor,
    foot: torch.Tensor,
    radius: torch.Tensor,
    iterations: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robustly refine a cylinder in its five identifiable local parameters."""
    axis = _unit(axis)
    foot = foot - axis * torch.dot(foot, axis)
    radius = radius.reshape(()).clamp_min(1e-8)
    spatial_scale = (points.max(dim=0).values - points.min(dim=0).values).norm()
    spatial_limit = spatial_scale.clamp_min(1e-3) * 0.25

    for _ in range(max(0, int(iterations))):
        offset = points - foot
        axial = offset @ axis
        radial_vector = offset - axial.unsqueeze(1) * axis
        radial = radial_vector.norm(dim=1).clamp_min(1e-10)
        residual = radial - radius
        basis = _orthogonal_basis(axis)
        axis_jacobian = (
            -(axial / radial).unsqueeze(1) * (offset @ basis)
        )
        center_jacobian = -(radial_vector / radial.unsqueeze(1)) @ basis
        jacobian = torch.cat(
            (axis_jacobian, center_jacobian, -torch.ones_like(radial).unsqueeze(1)),
            dim=1,
        )
        step = _damped_irls_step(jacobian, residual)
        if step is None:
            break
        axis_step = step[:2]
        if axis_step.norm() > 0.25:
            axis_step = axis_step * (0.25 / axis_step.norm())
        center_step = step[2:4]
        if center_step.norm() > spatial_limit:
            center_step = center_step * (spatial_limit / center_step.norm())
        radius_step = step[4].clamp(min=-spatial_limit, max=spatial_limit)
        current_score = _trimmed_residual_score(residual)
        accepted = False
        for factor in (1.0, 0.5, 0.25, 0.125):
            candidate_axis = _unit(axis + factor * (basis @ axis_step))
            candidate_foot = foot + factor * (basis @ center_step)
            candidate_foot = candidate_foot - candidate_axis * torch.dot(
                candidate_foot, candidate_axis
            )
            candidate_radius = (radius + factor * radius_step).clamp_min(1e-8)
            candidate_residual = _cylinder_residual(
                points, candidate_axis, candidate_foot, candidate_radius
            )
            if _trimmed_residual_score(candidate_residual) <= current_score:
                axis, foot, radius = (
                    candidate_axis,
                    candidate_foot,
                    candidate_radius,
                )
                accepted = True
                break
        if not accepted or float(step.norm().item()) < 1e-8:
            break
    return axis, radius, foot


def _cone_residual(
    points: torch.Tensor,
    axis: torch.Tensor,
    apex: torch.Tensor,
    semi_angle: torch.Tensor,
) -> torch.Tensor:
    offset = points - apex
    axial = offset @ axis
    radial = (offset - axial.unsqueeze(1) * axis).norm(dim=1)
    return radial - axial.abs() * torch.tan(semi_angle)


def _refine_cone_parameters(
    points: torch.Tensor,
    axis: torch.Tensor,
    apex: torch.Tensor,
    semi_angle: torch.Tensor,
    iterations: int = 15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robustly refine cone axis, apex, and semi-angle from an initial guess."""
    axis = _unit(axis)
    semi_angle = semi_angle.reshape(()).clamp(min=1e-4, max=1.55)
    spatial_scale = (points.max(dim=0).values - points.min(dim=0).values).norm()
    spatial_limit = spatial_scale.clamp_min(1e-3) * 0.25

    for _ in range(max(0, int(iterations))):
        offset = points - apex
        axial = offset @ axis
        axial_sign = torch.where(axial >= 0, 1.0, -1.0)
        radial_vector = offset - axial.unsqueeze(1) * axis
        radial = radial_vector.norm(dim=1).clamp_min(1e-10)
        tangent = torch.tan(semi_angle)
        residual = radial - axial.abs() * tangent
        basis = _orthogonal_basis(axis)
        projected_offset = offset @ basis
        axis_factor = -(axial / radial) - axial_sign * tangent
        axis_jacobian = axis_factor.unsqueeze(1) * projected_offset
        apex_perpendicular_jacobian = -(radial_vector / radial.unsqueeze(1)) @ basis
        apex_axial_jacobian = (axial_sign * tangent).unsqueeze(1)
        angle_jacobian = (
            -axial.abs() / torch.cos(semi_angle).square().clamp_min(1e-8)
        ).unsqueeze(1)
        jacobian = torch.cat(
            (
                axis_jacobian,
                apex_perpendicular_jacobian,
                apex_axial_jacobian,
                angle_jacobian,
            ),
            dim=1,
        )
        step = _damped_irls_step(jacobian, residual)
        if step is None:
            break
        axis_step = step[:2]
        if axis_step.norm() > 0.25:
            axis_step = axis_step * (0.25 / axis_step.norm())
        apex_perpendicular_step = step[2:4]
        if apex_perpendicular_step.norm() > spatial_limit:
            apex_perpendicular_step = apex_perpendicular_step * (
                spatial_limit / apex_perpendicular_step.norm()
            )
        apex_axial_step = step[4].clamp(min=-spatial_limit, max=spatial_limit)
        angle_step = step[5].clamp(min=-0.15, max=0.15)
        current_score = _trimmed_residual_score(residual)
        accepted = False
        for factor in (1.0, 0.5, 0.25, 0.125):
            candidate_axis = _unit(axis + factor * (basis @ axis_step))
            candidate_apex = apex + factor * (
                basis @ apex_perpendicular_step + axis * apex_axial_step
            )
            candidate_angle = (semi_angle + factor * angle_step).clamp(
                min=1e-4, max=1.55
            )
            candidate_residual = _cone_residual(
                points, candidate_axis, candidate_apex, candidate_angle
            )
            if _trimmed_residual_score(candidate_residual) <= current_score:
                axis, apex, semi_angle = (
                    candidate_axis,
                    candidate_apex,
                    candidate_angle,
                )
                accepted = True
                break
        if not accepted or float(step.norm().item()) < 1e-8:
            break
    return axis, semi_angle, apex


def _fit_plane(points: torch.Tensor, normals: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center, eigvec = _pca(points)
    pca_normal = eigvec[:, 0]
    normal = pca_normal
    if normals is not None and normals.numel() > 0:
        normal_mean = F.normalize(normals.mean(dim=0), dim=0, eps=1e-6)
        if torch.isfinite(normal_mean).all() and normal_mean.norm() > 0.1:
            if torch.dot(normal_mean, pca_normal) < 0:
                pca_normal = -pca_normal
            normal = F.normalize(0.5 * pca_normal + 0.5 * normal_mean, dim=0, eps=1e-6)
    normal = canonicalize_directions(normal.view(1, 3)).view(3)
    foot = normal * torch.dot(normal, center)
    dim = points.new_zeros(())
    return normal, dim, foot


def _fit_cylinder(
    points: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
    *,
    initial_direction: Optional[torch.Tensor] = None,
    initial_dimension: Optional[torch.Tensor] = None,
    initial_location: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_dtype = points.dtype
    work_dtype = torch.float64
    points = points.to(dtype=work_dtype)
    normals = None if normals is None else normals.to(dtype=work_dtype)
    initial_direction = (
        None if initial_direction is None else initial_direction.to(points).reshape(3)
    )
    initial_dimension = (
        None if initial_dimension is None else initial_dimension.to(points).reshape(())
    )
    initial_location = (
        None if initial_location is None else initial_location.to(points).reshape(3)
    )

    center, eigvec = _pca(points)
    pca_axis = eigvec[:, -1]
    axis_candidates: list[torch.Tensor] = []

    def append_axis(candidate: Optional[torch.Tensor]) -> None:
        if candidate is None or not torch.isfinite(candidate).all() or candidate.norm() < 1e-6:
            return
        candidate = _unit(candidate)
        if not any(torch.dot(candidate, existing).abs() > 1.0 - 1e-6 for existing in axis_candidates):
            axis_candidates.append(candidate)

    # Joint-head predictions are deliberately considered first. They seed the
    # optimizer but still have to win against coordinate-derived candidates.
    append_axis(initial_direction)
    normalized_normals = None
    if normals is not None and normals.shape[0] >= 3:
        normalized_normals = F.normalize(normals, dim=-1, eps=1e-6)
        cov = normalized_normals.transpose(0, 1) @ normalized_normals / float(
            normalized_normals.shape[0]
        )
        _, n_eigvec = _symmetric_eigh(cov)
        normal_axis = n_eigvec[:, 0]
        append_axis(normal_axis)
    append_axis(pca_axis)

    candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if (
        initial_direction is not None
        and initial_location is not None
        and initial_dimension is not None
        and torch.isfinite(initial_location).all()
        and torch.isfinite(initial_dimension)
        and float(initial_dimension.item()) > 0
    ):
        initial_axis = _unit(initial_direction)
        initial_foot = initial_location - initial_axis * torch.dot(
            initial_location, initial_axis
        )
        candidates.append((initial_axis, initial_dimension, initial_foot))

    for axis in axis_candidates:
        center_candidates: list[torch.Tensor] = []
        if normalized_normals is not None:
            normal_center, normal_basis = _fit_radial_center_2d(
                points, normalized_normals, axis
            )
            if normal_center is not None:
                center_candidates.append(normal_basis @ normal_center)
        circle_center, circle_basis = _fit_circle_center_2d(points, axis)
        if circle_center is not None:
            center_candidates.append(circle_basis @ circle_center)
        center_candidates.append(center - axis * torch.dot(axis, center))

        for foot in center_candidates:
            radial = torch.cross(
                points - foot, axis.expand_as(points), dim=1
            ).norm(dim=1)
            radius = radial.median() if radial.numel() >= 3 else radial.mean()
            if torch.isfinite(radius) and torch.isfinite(foot).all():
                candidates.append((axis, radius, foot))

    best = None
    for axis, radius, foot in candidates:
        refined_axis, refined_radius, refined_foot = _refine_cylinder_parameters(
            points, axis, foot, radius
        )
        residual = _cylinder_residual(
            points, refined_axis, refined_foot, refined_radius
        )
        score = _trimmed_residual_score(residual)
        if torch.isfinite(score) and (
            best is None or float(score.item()) < best[0]
        ):
            best = (
                float(score.item()), refined_axis, refined_radius, refined_foot
            )

    if best is None:
        axis = _unit(pca_axis)
        foot = center - axis * torch.dot(axis, center)
        radial = torch.cross(points - foot, axis.expand_as(points), dim=1).norm(dim=1)
        radius = _robust_mean(radial, trim_ratio=0.1)
    else:
        _, axis, radius, foot = best
    axis = canonicalize_directions(axis.view(1, 3)).view(3)
    return (
        axis.to(dtype=output_dtype),
        radius.to(dtype=output_dtype),
        foot.to(dtype=output_dtype),
    )


def _fit_cone(
    points: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
    *,
    initial_direction: Optional[torch.Tensor] = None,
    initial_dimension: Optional[torch.Tensor] = None,
    initial_location: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_dtype = points.dtype
    work_dtype = torch.float64
    points = points.to(dtype=work_dtype)
    normals = None if normals is None else normals.to(dtype=work_dtype)
    initial_direction = (
        None if initial_direction is None else initial_direction.to(points).reshape(3)
    )
    initial_dimension = (
        None if initial_dimension is None else initial_dimension.to(points).reshape(())
    )
    initial_location = (
        None if initial_location is None else initial_location.to(points).reshape(3)
    )

    center, eigvec = _pca(points)
    pca_axis = eigvec[:, -1]
    axis_candidates: list[torch.Tensor] = []

    def append_axis(candidate: Optional[torch.Tensor]) -> None:
        if candidate is None or not torch.isfinite(candidate).all() or candidate.norm() < 1e-6:
            return
        candidate = _unit(candidate)
        if not any(torch.dot(candidate, existing).abs() > 1.0 - 1e-6 for existing in axis_candidates):
            axis_candidates.append(candidate)

    append_axis(initial_direction)
    oriented_normals = None
    if normals is not None and normals.shape[0] >= 4:
        oriented_normals = _orient_normals_consistently(points, normals)
        centered_normals = oriented_normals - oriented_normals.mean(
            dim=0, keepdim=True
        )
        normal_cov = centered_normals.transpose(0, 1) @ centered_normals
        normal_cov = normal_cov / float(centered_normals.shape[0])
        _, normal_eigvec = _symmetric_eigh(normal_cov)
        normal_axis = normal_eigvec[:, 0]
        append_axis(normal_axis)
    append_axis(pca_axis)

    candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if (
        initial_direction is not None
        and initial_location is not None
        and initial_dimension is not None
        and torch.isfinite(initial_location).all()
        and torch.isfinite(initial_dimension)
        and 1e-4 <= float(initial_dimension.item()) <= 1.55
    ):
        candidates.append(
            (_unit(initial_direction), initial_dimension, initial_location)
        )

    for axis in axis_candidates:
        center_2d = None
        basis = _orthogonal_basis(axis)
        if oriented_normals is not None:
            center_2d, basis = _fit_radial_center_2d(
                points, oriented_normals, axis
            )
        if center_2d is None:
            center_2d = (points @ basis).mean(dim=0)

        point_2d = points @ basis
        axial = points @ axis
        profile = _refine_cone_profile(point_2d, axial, center_2d)
        if profile is not None:
            profile_center, slope, intercept = profile
            if float(slope.abs().item()) > 1e-4:
                apex_axial = -intercept / slope
                apex = basis @ profile_center + apex_axial * axis
                semi_angle = torch.atan(slope.abs()).clamp(max=1.55)
                if torch.isfinite(apex).all() and torch.isfinite(semi_angle):
                    candidates.append((axis, semi_angle, apex))

        # Finite fallback for near-cylindrical or otherwise degenerate profiles.
        projection = (points - center) @ axis
        fallback_apex = center + projection.min() * axis
        offset = points - fallback_apex
        signed_axial = offset @ axis
        axial_distance = signed_axial.abs()
        radial = (offset - signed_axial.unsqueeze(1) * axis).norm(dim=1)
        valid = axial_distance > 1e-4
        fallback_angle = (
            torch.atan((radial[valid] / axial_distance[valid]).median().clamp_min(0.0))
            if valid.any()
            else points.new_tensor(1e-4)
        )
        candidates.append((axis, fallback_angle.clamp(min=1e-4, max=1.55), fallback_apex))

    best = None
    for axis, semi_angle, apex in candidates:
        refined_axis, refined_angle, refined_apex = _refine_cone_parameters(
            points, axis, apex, semi_angle
        )
        residual = _cone_residual(
            points, refined_axis, refined_apex, refined_angle
        )
        score = _trimmed_residual_score(residual)
        if torch.isfinite(score) and (
            best is None or float(score.item()) < best[0]
        ):
            best = (
                float(score.item()), refined_axis, refined_angle, refined_apex
            )

    if best is None:
        axis = _unit(pca_axis)
        apex = center
        semi_angle = points.new_tensor(1e-4)
    else:
        _, axis, semi_angle, apex = best
    axis = canonicalize_directions(axis.view(1, 3)).view(3)
    return (
        axis.to(dtype=output_dtype),
        semi_angle.to(dtype=output_dtype),
        apex.to(dtype=output_dtype),
    )


def _fit_sphere(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if points.shape[0] >= 4:
        a_mat = torch.cat([2.0 * points, torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)], dim=1)
        rhs = (points * points).sum(dim=1, keepdim=True)
        try:
            sol = _least_squares(a_mat, rhs).squeeze(1)
            center = sol[:3]
            radius_sq = (sol[3] + (center * center).sum()).clamp_min(0.0)
            radius = torch.sqrt(radius_sq)
            if not (torch.isfinite(center).all() and torch.isfinite(radius)):
                raise RuntimeError("non-finite sphere fit")
        except RuntimeError:
            center = points.mean(dim=0)
            radius = (points - center).norm(dim=1).mean()
    else:
        center = points.mean(dim=0)
        radius = (points - center).norm(dim=1).mean()
    direction = points.new_tensor(INVALID_DIRECTION)
    return direction, radius, center


def _fit_other(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return points.new_tensor(INVALID_DIRECTION), points.new_zeros(()), points.new_zeros(3)


def _majority(values: torch.Tensor, n_classes: int = N_PRIMITIVES) -> int:
    if values.numel() == 0:
        return n_classes - 1
    return int(torch.bincount(values.long(), minlength=n_classes).argmax().item())


def _aggregate_axis_prediction(values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Aggregate sign-ambiguous per-point axes with an outer-product mean."""
    if values is None or values.numel() == 0:
        return None
    valid = torch.isfinite(values).all(dim=1) & (values.norm(dim=1) > 1e-6)
    if not bool(valid.any()):
        return None
    unit_values = F.normalize(values[valid].float(), dim=1, eps=1e-6)
    scatter = unit_values.transpose(0, 1) @ unit_values
    _, eigenvectors = _symmetric_eigh(scatter)
    axis = eigenvectors[:, -1].to(dtype=values.dtype)
    return canonicalize_directions(axis.view(1, 3)).view(3)


def _aggregate_scalar_prediction(values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if values is None or values.numel() == 0:
        return None
    valid = torch.isfinite(values)
    if not bool(valid.any()):
        return None
    return values[valid].median()


def _aggregate_location_prediction(values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if values is None or values.numel() == 0:
        return None
    valid = torch.isfinite(values).all(dim=1)
    if not bool(valid.any()):
        return None
    return values[valid].median(dim=0).values


def assemble_constraints_from_stage1(
    xyz: torch.Tensor,
    cluster_embedding: torch.Tensor,
    log_primitive: torch.Tensor,
    cluster_bandwidth: float = 0.35,
    normal_k: int = 16,
    use_robust_fitting: bool = True,
    use_pca_normals_for_fitting: bool = True,
    mad_prediction: Optional[torch.Tensor] = None,
    dim_prediction: Optional[torch.Tensor] = None,
    loc_prediction: Optional[torch.Tensor] = None,
    use_prediction_initialization: bool = True,
    cluster_method: str = "radius",
    mean_shift_quantile: float = 0.015,
    mean_shift_iterations: int = 50,
    mean_shift_max_clusters: int = 128,
    mean_shift_bandwidth: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert Stage 1 primitive logits and clustering embeddings to constraints.

    xyz: [B, N, 3]
    cluster_embedding: [B, N, D]
    log_primitive: [B, N, 5]

    PCA normals are an optional fitting-only intermediate for plane, cylinder,
    and cone clusters. Joint-head geometry predictions are robustly aggregated
    per cluster and used only to initialize cylinder/cone fitting; final values
    still come from the XYZ fit.
    """
    if cluster_method not in CLUSTER_METHODS:
        raise ValueError(
            f"unsupported cluster_method={cluster_method!r}; expected one of {CLUSTER_METHODS}"
        )
    bsz, n_points, _ = xyz.shape
    device = xyz.device
    dtype = xyz.dtype

    pmt_idx = log_primitive.argmax(dim=-1)
    primitive_type = torch.zeros(bsz, n_points, N_PRIMITIVES, device=device, dtype=dtype)
    direction = torch.zeros(bsz, n_points, 3, device=device, dtype=dtype)
    dimension = torch.zeros(bsz, n_points, device=device, dtype=dtype)
    location = torch.zeros(bsz, n_points, 3, device=device, dtype=dtype)
    affiliate_idx = torch.full(
        (bsz, n_points), -1, device=device, dtype=torch.long
    )

    for b in range(bsz):
        if cluster_method == "meanshift":
            labels = cluster_embeddings_mean_shift(
                cluster_embedding[b],
                quantile=mean_shift_quantile,
                iterations=mean_shift_iterations,
                max_clusters=mean_shift_max_clusters,
                bandwidth=mean_shift_bandwidth,
            ).to(device)
        else:
            labels = cluster_embeddings_radius(
                cluster_embedding[b], bandwidth=cluster_bandwidth
            ).to(device)
        affiliate_idx[b] = labels
        for cluster_id in labels.unique(sorted=True):
            mask = labels == cluster_id
            points = xyz[b, mask]
            prim = _majority(pmt_idx[b, mask])
            primitive_type[b, mask, prim] = 1.0
            initial_direction = initial_dimension = initial_location = None
            if use_prediction_initialization:
                initial_direction = _aggregate_axis_prediction(
                    None if mad_prediction is None else mad_prediction[b, mask]
                )
                initial_dimension = _aggregate_scalar_prediction(
                    None if dim_prediction is None else dim_prediction[b, mask]
                )
                initial_location = _aggregate_location_prediction(
                    None if loc_prediction is None else loc_prediction[b, mask]
                )
            cluster_normals = None
            if (
                use_robust_fitting
                and use_pca_normals_for_fitting
                and prim in (0, 1, 2)
            ):
                cluster_normals = estimate_normals_pca(
                    points.unsqueeze(0), k=normal_k
                )[0].to(dtype=dtype)

            if prim == 0:
                fit_dir, fit_dim, fit_loc = _fit_plane(points, cluster_normals)
            elif prim == 1:
                fit_dir, fit_dim, fit_loc = _fit_cylinder(
                    points,
                    cluster_normals,
                    initial_direction=initial_direction,
                    initial_dimension=initial_dimension,
                    initial_location=initial_location,
                )
            elif prim == 2:
                fit_dir, fit_dim, fit_loc = _fit_cone(
                    points,
                    cluster_normals,
                    initial_direction=initial_direction,
                    initial_dimension=initial_dimension,
                    initial_location=initial_location,
                )
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
        "location": location,
        "affiliate_idx": affiliate_idx,
    }


def constraints_to_tensor(constraints: Dict[str, torch.Tensor]) -> torch.Tensor:
    primitive_type = constraints["primitive_type"]
    primitive_index = primitive_type.argmax(dim=-1)
    direction, dimension = zero_invalid_constraint_components(
        primitive_index,
        constraints["direction"],
        constraints["dimension"],
    )
    return torch.cat(
        [
            primitive_type,
            direction,
            dimension.unsqueeze(-1)
            if dimension.dim() == primitive_index.dim()
            else dimension,
            constraints["location"],
        ],
        dim=-1,
    )


def ground_truth_constraints_to_tensor(
    pmt: torch.Tensor,
    direction: torch.Tensor,
    dimension: torch.Tensor,
    location: torch.Tensor,
    n_primitives: int = N_PRIMITIVES,
) -> torch.Tensor:
    pmt_one_hot = F.one_hot(pmt.long(), n_primitives).to(dtype=direction.dtype, device=direction.device)
    direction, dimension = zero_invalid_constraint_components(
        pmt, direction, dimension
    )
    return torch.cat(
        [
            pmt_one_hot,
            direction,
            dimension.unsqueeze(-1) if dimension.dim() == pmt.dim() else dimension,
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
        "location": constraints[..., 9:12],
    }
