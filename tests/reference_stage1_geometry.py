"""Frozen pre-vectorization geometry losses, used only as a numerical test oracle."""
import torch
import torch.nn.functional as F
from functional.loss import (_zero_loss, _primitive_mask, PARAMETER_SMOOTH_L1_BETA,
    GEOMETRY_SMOOTH_L1_BETA, LARGE_PARAMETER_RATIO_THRESHOLD, MIN_OBSERVABILITY_WEIGHT)

def _masked_mean(values: torch.Tensor, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return _zero_loss(reference)


def _smooth_l1_zero(values: torch.Tensor, beta: float) -> torch.Tensor:
    """Element-wise Smooth L1 residual with a zero target."""
    return F.smooth_l1_loss(
        values,
        torch.zeros_like(values),
        reduction="none",
        beta=beta,
    )


def _instance_balanced_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    affiliate_idx: torch.Tensor,
    reference: torch.Tensor,
    instance_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average point losses inside each primitive, then average primitives.

    Primitive parameters are repeated at every sampled point. A plain point
    mean therefore gives densely sampled or large faces more influence even
    though they carry only one parameter tuple. This reduction gives every
    visible primitive instance one term instead.
    """
    if values.shape != mask.shape or affiliate_idx.shape != mask.shape:
        raise ValueError("instance-balanced loss inputs must all have shape [B, N]")
    if instance_weights is not None and instance_weights.shape != mask.shape:
        raise ValueError("instance_weights must have shape [B, N]")
    if not bool(mask.any()):
        return _zero_loss(reference)

    batch_ids = torch.arange(
        mask.shape[0], device=mask.device, dtype=affiliate_idx.dtype
    ).unsqueeze(1).expand_as(affiliate_idx)
    keys = torch.stack(
        (batch_ids[mask], affiliate_idx[mask]), dim=-1
    )
    unique_keys, inverse = torch.unique(
        keys, dim=0, sorted=True, return_inverse=True
    )
    instance_count = unique_keys.shape[0]
    sums = torch.zeros(
        instance_count, device=values.device, dtype=values.dtype
    )
    sums.index_add_(0, inverse, values[mask])
    counts = torch.bincount(inverse, minlength=instance_count).to(values.dtype)
    terms = sums / counts.clamp_min(1.0)
    if instance_weights is not None:
        weight_sums = torch.zeros_like(sums)
        weight_sums.index_add_(0, inverse, instance_weights[mask].to(values.dtype))
        # Do not renormalize the final terms by the sum of weights: a
        # low-confidence primitive must genuinely contribute less, including
        # when it is the only primitive of that type.
        terms = terms * (weight_sums / counts.clamp_min(1.0))
    return terms.mean()


@torch.no_grad()
def _parameter_observability_weights(
    xyz: torch.Tensor,
    pmt_gt: torch.Tensor,
    mad_gt: torch.Tensor,
    dim_gt: torch.Tensor,
    loc_gt: torch.Tensor,
    affiliate_idx: torch.Tensor,
    ratio_threshold: float = LARGE_PARAMETER_RATIO_THRESHOLD,
    min_weight: float = MIN_OBSERVABILITY_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft confidence weights for weakly observable DIM and LOC labels.

    A radius or remote center/axis/apex that is many times larger than the
    observed patch is intrinsically ill-conditioned. The weight decreases
    smoothly beyond ``ratio_threshold`` but never drops below ``min_weight``.
    Cylinder patch size is measured perpendicular to its GT axis so a long,
    narrow cylindrical strip is not incorrectly considered well observed.
    """
    if ratio_threshold <= 0:
        raise ValueError("ratio_threshold must be positive")
    if not 0 < min_weight <= 1:
        raise ValueError("min_weight must be in (0, 1]")

    batch_ids = torch.arange(
        xyz.shape[0], device=xyz.device, dtype=affiliate_idx.dtype
    ).unsqueeze(1).expand_as(affiliate_idx)
    keys = torch.stack(
        (batch_ids.reshape(-1), affiliate_idx.reshape(-1)), dim=-1
    )
    unique_keys, inverse = torch.unique(
        keys, dim=0, sorted=True, return_inverse=True
    )
    instance_count = unique_keys.shape[0]
    counts = torch.bincount(inverse, minlength=instance_count).float().clamp_min(1.0)

    xyz_flat = xyz.reshape(-1, 3).float()
    xyz_sums = torch.zeros(
        instance_count, 3, device=xyz.device, dtype=torch.float32
    )
    xyz_sums.index_add_(0, inverse, xyz_flat)
    xyz_centers = xyz_sums / counts.unsqueeze(-1)
    centered = xyz_flat - xyz_centers[inverse]

    # Project each cylinder point using its own GT axis. The projection is
    # unchanged for axis v or -v and therefore preserves the direction fix.
    direction_flat = F.normalize(
        mad_gt.reshape(-1, 3).float(), dim=-1, eps=1e-6
    )
    transverse = centered - (
        centered * direction_flat
    ).sum(dim=-1, keepdim=True) * direction_flat
    primitive_flat = pmt_gt.reshape(-1).long()
    extent_vectors = torch.where(
        (primitive_flat == 1).unsqueeze(-1), transverse, centered
    )
    point_extents = extent_vectors.norm(dim=-1)
    max_extents = torch.zeros(
        instance_count, device=xyz.device, dtype=torch.float32
    )
    max_extents.scatter_reduce_(
        0, inverse, point_extents, reduce="amax", include_self=True
    )
    patch_diameters = (2.0 * max_extents).clamp_min(1e-3)

    primitive_counts = torch.zeros(
        instance_count * 5, device=xyz.device, dtype=torch.float32
    )
    primitive_counts.index_add_(
        0,
        inverse * 5 + primitive_flat.clamp(min=0, max=4),
        torch.ones_like(primitive_flat, dtype=torch.float32),
    )
    instance_primitives = primitive_counts.reshape(instance_count, 5).argmax(dim=-1)

    dim_sums = torch.zeros_like(max_extents)
    dim_sums.index_add_(0, inverse, dim_gt.reshape(-1).float().abs())
    radius_scales = dim_sums / counts

    loc_sums = torch.zeros_like(xyz_sums)
    loc_sums.index_add_(0, inverse, loc_gt.reshape(-1, 3).float())
    location_scales = (loc_sums / counts.unsqueeze(-1) - xyz_centers).norm(dim=-1)

    def confidence(parameter_scales: torch.Tensor) -> torch.Tensor:
        return (
            ratio_threshold * patch_diameters
            / parameter_scales.clamp_min(1e-6)
        ).clamp(min=min_weight, max=1.0)

    dim_instance_weights = torch.where(
        (instance_primitives == 1) | (instance_primitives == 3),
        confidence(radius_scales),
        torch.ones_like(radius_scales),
    )
    # Plane offsets remain directly observable. Remote cylinder axes, cone
    # apices, and sphere centers receive a soft confidence weight.
    loc_instance_weights = torch.where(
        (instance_primitives == 1)
        | (instance_primitives == 2)
        | (instance_primitives == 3),
        confidence(location_scales),
        torch.ones_like(location_scales),
    )
    return (
        dim_instance_weights[inverse].reshape_as(dim_gt).to(xyz.dtype),
        loc_instance_weights[inverse].reshape_as(dim_gt).to(xyz.dtype),
    )


def _robust_dimension_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pmt_gt: torch.Tensor,
    affiliate_idx: torch.Tensor,
    observability_weights: torch.Tensor,
) -> torch.Tensor:
    radius_mask = _primitive_mask(pmt_gt, (1, 3))
    cone_mask = pmt_gt == 2
    valid_mask = radius_mask | cone_mask

    # Radius is positive and may span orders of magnitude. log1p preserves
    # resolution near zero while turning large absolute errors into scale-like
    # errors. Cone semi-angle is already bounded and stays in radians.
    pred_radius = torch.log1p(pred.clamp_min(0.0))
    target_radius = torch.log1p(target.clamp_min(0.0))
    radius_errors = F.smooth_l1_loss(
        pred_radius,
        target_radius,
        reduction="none",
        beta=PARAMETER_SMOOTH_L1_BETA,
    )
    cone_errors = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=PARAMETER_SMOOTH_L1_BETA,
    )
    point_errors = torch.where(
        radius_mask,
        radius_errors,
        torch.where(cone_mask, cone_errors, torch.zeros_like(pred)),
    )
    return _instance_balanced_mean(
        point_errors,
        valid_mask,
        affiliate_idx,
        pred,
        instance_weights=observability_weights,
    )


def _robust_location_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pmt_gt: torch.Tensor,
    affiliate_idx: torch.Tensor,
    observability_weights: torch.Tensor,
) -> torch.Tensor:
    valid_mask = _primitive_mask(pmt_gt, (0, 1, 2, 3))
    point_errors = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=PARAMETER_SMOOTH_L1_BETA,
    ).mean(dim=-1)
    return _instance_balanced_mean(
        point_errors,
        valid_mask,
        affiliate_idx,
        pred,
        instance_weights=observability_weights,
    )


def _masked_vector_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sign_invariant: bool = False,
) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(pred)
    pred_m = F.normalize(pred[mask], dim=-1, eps=1e-6)
    target_m = F.normalize(target[mask], dim=-1, eps=1e-6)
    if sign_invariant:
        # Primitive directions are unoriented axes. Compare both equivalent
        # representatives directly instead of relying on a discontinuous
        # world-axis sign convention during optimization.
        direct = (pred_m - target_m).pow(2).mean(dim=-1)
        flipped = (pred_m + target_m).pow(2).mean(dim=-1)
        return torch.minimum(direct, flipped).mean()
    return F.mse_loss(pred_m, target_m)


def _masked_scalar_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(pred)
    return F.mse_loss(pred[mask], target[mask])


def _masked_parallel_loss(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    mask: torch.Tensor,
    affiliate_idx: torch.Tensor | None = None,
    instance_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(vec1)
    a = F.normalize(vec1, dim=-1, eps=1e-6)
    b = F.normalize(vec2, dim=-1, eps=1e-6)
    values = (1.0 - (a * b).sum(dim=-1).abs()).pow(2)
    if affiliate_idx is None:
        return values[mask].mean()
    return _instance_balanced_mean(
        values, mask, affiliate_idx, vec1, instance_weights
    )


def _masked_perpendicular_loss(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    mask: torch.Tensor,
    affiliate_idx: torch.Tensor | None = None,
    instance_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(vec1)
    a = F.normalize(vec1, dim=-1, eps=1e-6)
    b = F.normalize(vec2, dim=-1, eps=1e-6)
    values = (a * b).sum(dim=-1).pow(2)
    if affiliate_idx is None:
        return values[mask].mean()
    return _instance_balanced_mean(
        values, mask, affiliate_idx, vec1, instance_weights
    )


def _stage1_geometry_losses(
    xyz: torch.Tensor,
    mad_pred: torch.Tensor,
    dim_pred: torch.Tensor,
    loc_pred: torch.Tensor,
    pmt_gt: torch.Tensor,
    affiliate_idx: torch.Tensor,
    dim_observability: torch.Tensor,
    loc_observability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mad_pred = F.normalize(mad_pred, dim=-1, eps=1e-6)
    dim_pred = dim_pred.clamp_min(0.0)

    plane_mask = pmt_gt == 0
    if plane_mask.any():
        n = mad_pred
        plane_residual = ((xyz - loc_pred) * n).sum(dim=-1)
        plane_error = _smooth_l1_zero(
            plane_residual, GEOMETRY_SMOOTH_L1_BETA
        )
        on_plane = _instance_balanced_mean(
            plane_error, plane_mask, affiliate_idx, xyz
        )
        loc_nonzero = plane_mask & (loc_pred.norm(dim=-1) > 1e-4)
        loc_parallel = _masked_parallel_loss(
            loc_pred, mad_pred, loc_nonzero, affiliate_idx
        )
        loss_plane = on_plane + 0.2 * loc_parallel
    else:
        loss_plane = _zero_loss(xyz)

    cylinder_mask = pmt_gt == 1
    if cylinder_mask.any():
        v = xyz - loc_pred
        axis = mad_pred
        axial = (v * axis).sum(dim=-1, keepdim=True) * axis
        radial = (v - axial).norm(dim=-1)
        radius_error = _smooth_l1_zero(
            radial - dim_pred, GEOMETRY_SMOOTH_L1_BETA
        )
        cylinder_observability = torch.minimum(
            dim_observability, loc_observability
        )
        on_cylinder = _instance_balanced_mean(
            radius_error,
            cylinder_mask,
            affiliate_idx,
            xyz,
            cylinder_observability,
        )
        loc_perp_axis = _masked_perpendicular_loss(
            loc_pred,
            mad_pred,
            cylinder_mask,
            affiliate_idx,
            cylinder_observability,
        )
        loss_cylinder = on_cylinder + 0.2 * loc_perp_axis
    else:
        loss_cylinder = _zero_loss(xyz)

    cone_mask = pmt_gt == 2
    if cone_mask.any():
        v = xyz - loc_pred
        axis = mad_pred
        signed_axial = (v * axis).sum(dim=-1)
        radial_vec = v - signed_axial.unsqueeze(-1) * axis
        radial = radial_vec.norm(dim=-1)
        distance_to_apex = v.norm(dim=-1).clamp_min(1e-4)
        semi_angle = dim_pred.clamp(min=1e-4, max=1.55)
        # Compare normalized radial direction with sin(semi_angle). Unlike
        # radial - axial*tan(angle), this residual and its angle derivative do
        # not explode for a remote apex or an angle close to pi/2.
        cone_error = _smooth_l1_zero(
            radial / distance_to_apex - torch.sin(semi_angle),
            GEOMETRY_SMOOTH_L1_BETA,
        )
        loss_cone = _instance_balanced_mean(
            cone_error,
            cone_mask,
            affiliate_idx,
            xyz,
            loc_observability,
        )
    else:
        loss_cone = _zero_loss(xyz)

    sphere_mask = pmt_gt == 3
    if sphere_mask.any():
        center_to_xyz = xyz - loc_pred
        radius_error = _smooth_l1_zero(
            center_to_xyz.norm(dim=-1) - dim_pred,
            GEOMETRY_SMOOTH_L1_BETA,
        )
        sphere_observability = torch.minimum(
            dim_observability, loc_observability
        )
        loss_sphere = _instance_balanced_mean(
            radius_error,
            sphere_mask,
            affiliate_idx,
            xyz,
            sphere_observability,
        )
    else:
        loss_sphere = _zero_loss(xyz)

    geom_loss = loss_plane + loss_cylinder + loss_cone + loss_sphere
    return {
        "geom_loss": geom_loss,
        "loss_plane": loss_plane,
        "loss_cylinder": loss_cylinder,
        "loss_cone": loss_cone,
        "loss_sphere": loss_sphere,
    }


def instance_consistency_loss(log_pmt_pred, mad_pred, dim_pred, loc_pred, affil_idx, pmt_gt=None):
    """
    log_pmt_pred: [B, P, 5] log-softmax后的基元类型预测
    mad_pred: [B, P, 3] 主方向预测
    dim_pred: [B, P] 尺寸预测
    loc_pred: [B, P, 3] 主位置预测
    affil_idx: [B, P] 每个点所属实例的索引 (int)
    pmt_gt: [B, P] optional primitive type labels used for valid property masks
    """
    bs = log_pmt_pred.size(0)
    terms = []
    mad_pred = F.normalize(mad_pred, dim=-1, eps=1e-6)
    probs = log_pmt_pred.exp()

    for b in range(bs):
        # 找到无重复的实例id
        inst_ids = affil_idx[b].unique()

        for inst_id in inst_ids:
            mask = (affil_idx[b] == inst_id)  # 当前实例的点
            if mask.sum() <= 1:
                continue  # 只有1个点不计算一致性

            if pmt_gt is None:
                inst_prim = None
            else:
                inst_labels = pmt_gt[b][mask].long()
                inst_prim = int(torch.bincount(inst_labels, minlength=5).argmax().item())

            # ---- 基元类型一致性（对logits取均值，再和每个点对齐）----
            pmt_prob = probs[b][mask]   # [N, 5]
            mean_prob = pmt_prob.mean(0, keepdim=True)  # [1, 5]
            terms.append(F.mse_loss(pmt_prob, mean_prob.expand_as(pmt_prob)))

            # ---- 主方向一致性 ----
            if inst_prim is None or inst_prim in (0, 1, 2):
                mad = mad_pred[b][mask]  # [N, 3]
                # Align signs relative to one detached member before averaging.
                # Thus v and -v reinforce the same cluster axis instead of
                # cancelling each other around the dir_unify boundary.
                reference = mad[:1].detach()
                flip = (mad * reference).sum(dim=-1, keepdim=True) < 0
                aligned_mad = torch.where(flip, -mad, mad)
                mean_mad = F.normalize(
                    aligned_mad.mean(0, keepdim=True), dim=-1, eps=1e-6
                )
                terms.append(
                    F.mse_loss(aligned_mad, mean_mad.expand_as(aligned_mad))
                )

            # ---- 尺寸一致性 ----
            if inst_prim is None or inst_prim in (1, 2, 3):
                dim = dim_pred[b][mask]  # [N]
                if inst_prim in (1, 3):
                    dim = torch.log1p(dim.clamp_min(0.0))
                mean_dim = dim.mean()
                terms.append(
                    F.smooth_l1_loss(
                        dim,
                        mean_dim.expand_as(dim),
                        beta=PARAMETER_SMOOTH_L1_BETA,
                    )
                )

            # ---- 主位置一致性 ----
            if inst_prim is None or inst_prim in (0, 1, 2, 3):
                loc = loc_pred[b][mask]  # [N, 3]
                mean_loc = loc.mean(0, keepdim=True)
                terms.append(
                    F.smooth_l1_loss(
                        loc,
                        mean_loc.expand_as(loc),
                        beta=PARAMETER_SMOOTH_L1_BETA,
                    )
                )

    if len(terms) == 0:
        return _zero_loss(log_pmt_pred)
    return torch.stack(terms).mean()

