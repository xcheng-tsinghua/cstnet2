import torch.nn.functional as F
import torch


def mse_loss_with_pmt_considered(attr_pred, attr_gt, pmt_gt, valid_pmt):
    """
    计算 attr_pred 和 attr_gt 之间的 mse_loss，只有有效类型的店才会参与计算

    :param attr_pred: [bs, point]
    :param attr_gt: [bs, point]
    :param pmt_gt: [bs, point] (int, index)
    :param valid_pmt: tuple: (1, 2, ...), (0=plane,1=cylinder,2=cone,3=sphere,4=freeform)
    :return:
    """
    # 筛选 mask：GT or 预测的基元类型是否在有效集合里
    mask = torch.isin(pmt_gt, torch.tensor(valid_pmt, device=pmt_gt.device))

    if mask.sum() > 0:

        # 只对有效类型计算 loss
        loss = F.mse_loss(attr_pred[mask], attr_gt[mask])
        return loss

    else:
        return 0.0


def unit_len_loss(attr_pred):
    """
    计算长度为1的loss
    :param attr_pred: [bs, ..., X]
    """
    # 计算每个向量的长度 (L2 norm)
    lengths = torch.norm(attr_pred, dim=-1)  # shape [bs]

    # 希望每个长度接近 1，可以用 MSE loss
    loss = F.mse_loss(lengths, torch.ones_like(lengths))
    return loss


def unit_len_loss_with_pmt_considered(attr_pred, pmt_gt, valid_pmt):
    """
    计算 attr_pred 长度与 1 之间的 loss，只有有效类型的店才会参与计算

    :param attr_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    :param valid_pmt: tuple: (1, 2, ...), (0=plane,1=cylinder,2=cone,3=sphere,4=freeform)
    :return:
    """
    # 筛选 mask：GT or 预测的基元类型是否在有效集合里
    mask = torch.isin(pmt_gt, torch.tensor(valid_pmt, device=pmt_gt.device))

    if mask.sum() > 0:
        valid_attr_pred = attr_pred[mask]
        loss = unit_len_loss(valid_attr_pred)
        return loss

    else:
        return 0.0


def perpendicular_loss(vec1, vec2, eps=1e-6):
    """
    计算对应位置向量垂直的 Loss
    :param vec1: [..., X]
    :param vec2: [..., X]
    :param eps: [..., X]
    """
    # 归一化（防止除零）
    a_norm = vec1 / (vec1.norm(dim=-1, keepdim=True) + eps)
    b_norm = vec2 / (vec2.norm(dim=-1, keepdim=True) + eps)

    # 点积
    dot = (a_norm * b_norm).sum(dim=-1)  # [bs, point]

    # loss: 点积平方的平均值
    loss = (dot ** 2).mean()
    return loss


def parallel_loss(vec1, vec2, eps=1e-6):
    # 归一化（防止除零）
    a_norm = vec1 / (vec1.norm(dim=-1, keepdim=True) + eps)
    b_norm = vec2 / (vec2.norm(dim=-1, keepdim=True) + eps)

    # 计算余弦相似度
    cos_sim = (a_norm * b_norm).sum(dim=-1)  # [-1, 1]

    # 平行时 |cos|=1
    loss = ((1 - cos_sim.abs()) ** 2).mean()
    return loss


def geom_loss_plane(xyz, mad_pred, nor_pred, loc_pred, pmt_gt):
    """
    :param xyz: [bs, point, 3]
    :param mad_pred: [bs, point, 3]
    :param nor_pred: [bs, point, 3]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部平面类型的点
    mask = (pmt_gt == 0)  # [bs, point]

    if mask.sum() > 0:

        xyz = xyz[mask]  # [n_item, 3]
        mad_pred = mad_pred[mask]  # [n_item, 3]
        nor_pred = nor_pred[mask]  # [n_item, 3]
        loc_pred = loc_pred[mask]  # [n_item, 3]

        # 点到预测平面的距离为 0，主要
        foot_to_xyz = xyz - loc_pred  # [bs, n, 3]
        nor_pred = nor_pred / (nor_pred.norm(dim=-1, keepdim=True) + 1e-8)
        dist = (foot_to_xyz * nor_pred).sum(dim=-1)
        on_plane_loss = (dist ** 2).mean()

        # 原点到垂足的向量与主方向平行，次要
        foot_pall_mad = parallel_loss(loc_pred, mad_pred)

        # 主方向和法线平行，次要
        mad_pall_nor = parallel_loss(mad_pred, nor_pred)

        return on_plane_loss + 0.2 * foot_pall_mad + 0.2 * mad_pall_nor

    else:
        return 0.0


def geom_loss_cylinder(xyz, mad_pred, nor_pred, dim_pred, loc_pred, pmt_gt):
    """
    :param xyz: [bs, point, 3]
    :param mad_pred: [bs, point, 3]
    :param nor_pred: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部圆柱类型的点
    mask = (pmt_gt == 1)  # [bs, point]

    if mask.sum() > 0:

        xyz = xyz[mask]  # [n_item, 3]
        mad_pred = mad_pred[mask]  # [n_item, 3]
        nor_pred = nor_pred[mask]  # [n_item, 3]
        dim_pred = dim_pred[mask]  # [n_item, 3]
        loc_pred = loc_pred[mask]  # [n_item, 3]

        # 半径与预测主尺寸相等，即点在圆柱上
        radius = torch.cross(xyz - loc_pred, mad_pred, dim=1)
        radius = radius.norm(dim=1)
        xyz_on_cylin = (radius - dim_pred).abs().mean()

        # 原点到垂足的向量与主方向垂直
        foot_prep_mad_loss = perpendicular_loss(loc_pred, mad_pred)
        # dot_product = torch.einsum('ij, ij -> i', loc_pred, mad_pred).abs().mean()

        # 法线与主方向垂直
        mad_prep_nor = perpendicular_loss(mad_pred, nor_pred)

        return xyz_on_cylin + 0.2 * foot_prep_mad_loss + 0.2 * mad_prep_nor

    else:
        return 0.0


# def geom_loss_cone(xyz, mad_pred, dim_pred, loc_pred, pmt_gt):
#     """
#     点在圆锥上
#     :param xyz: [bs, point, 3]
#     :param mad_pred: [bs, point, 3]
#     :param dim_pred: [bs, point]
#     :param loc_pred: [bs, point, 3]
#     :param pmt_gt: [bs, point] (int, index)
#     """
#     # 找到全部圆锥类型的点
#     mask = (pmt_gt == 2)  # [bs, point]
#
#     if mask.sum() > 0:
#
#         xyz = xyz[mask]  # [n_item, 3]
#         mad_pred = mad_pred[mask]  # [n_item, 3]
#         dim_pred = dim_pred[mask]  # [n_item, ]
#         loc_pred = loc_pred[mask]  # [n_item, 3]
#
#         # 从锥角到圆锥面上的点构成的向量与主方向之间的夹角等于主尺寸
#         apex_to_xyz = xyz - loc_pred
#         dot1 = torch.einsum('ij, ij -> i', mad_pred, apex_to_xyz)
#         dot2 = mad_pred * apex_to_xyz.norm(dim=1) * torch.cos(dim_pred)
#         semi_angle = (dot1 - dot2).abs().mean()
#
#         return semi_angle
#
#     else:
#         return 0.0


def geom_loss_cone(xyz, mad_pred, dim_pred, loc_pred, pmt_gt):
    """
    点在圆锥上
    :param xyz: [bs, point, 3]
    :param mad_pred: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部圆锥类型的点
    mask = (pmt_gt == 2)  # [bs, point]

    if mask.sum() > 0:

        mad_pred = mad_pred[mask]  # [n_item, 3]
        loc_pred = loc_pred[mask]  # [n_item, 3]

        # 原点到垂足的向量与主方向垂直
        foot_perp_mad = perpendicular_loss(loc_pred, mad_pred)
        # foot_perp_mad = torch.einsum('ij, ij -> i', loc_pred, mad_pred).abs().mean()

        return foot_perp_mad

    else:
        return 0.0


def geom_loss_sphere(xyz, dim_pred, nor_pred, loc_pred, pmt_gt):
    """
    :param xyz: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param nor_pred: [bs, point, 3]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部球类型的点
    mask = (pmt_gt == 3)  # [bs, point]

    if mask.sum() > 0:

        xyz = xyz[mask]  # [n_item, 3]
        dim_pred = dim_pred[mask]  # [n_item, ]
        nor_pred = nor_pred[mask]  # [n_item, 3]
        loc_pred = loc_pred[mask]  # [n_item, 3]

        # 球面上的点到主位置的距离等于主尺寸
        center_to_xyz = xyz - loc_pred
        xyz_on_sphere_loss = (center_to_xyz.norm(dim=1) - dim_pred).abs().mean()

        # 球心到 xyz 的向量与 nor 垂直
        center_to_xyz_pall_nor_loss = parallel_loss(center_to_xyz, nor_pred)

        return xyz_on_sphere_loss + 0.2 * center_to_xyz_pall_nor_loss

    else:
        return 0.0


def instance_consistency_loss(log_pmt_pred, mad_pred, dim_pred, loc_pred, affil_idx):
    """
    log_pmt_pred: [B, P, 5] log-softmax后的基元类型预测
    mad_pred: [B, P, 3] 主方向预测
    dim_pred: [B, P] 尺寸预测
    loc_pred: [B, P, 3] 主位置预测
    affil_idx: [B, P] 每个点所属实例的索引 (int)
    """
    bs = log_pmt_pred.size(0)
    loss_total = 0.0

    for b in range(bs):
        # 找到无重复的实例id
        inst_ids = affil_idx[b].unique()

        for inst_id in inst_ids:
            mask = (affil_idx[b] == inst_id)  # 当前实例的点
            if mask.sum() <= 1:
                continue  # 只有1个点不计算一致性

            # ---- 基元类型一致性（对logits取均值，再和每个点对齐）----
            logits = log_pmt_pred[b][mask]   # [N, 5]
            mean_logits = logits.mean(0, keepdim=True)  # [1, 5]
            loss_pmt = F.mse_loss(logits, mean_logits.expand_as(logits))

            # ---- 主方向一致性 ----
            mad = mad_pred[b][mask]  # [N, 3]
            mean_mad = mad.mean(0, keepdim=True)
            loss_mad = F.mse_loss(mad, mean_mad.expand_as(mad))

            # ---- 尺寸一致性 ----
            dim = dim_pred[b][mask]  # [N]
            mean_dim = dim.mean()
            loss_dim = F.mse_loss(dim, mean_dim.expand_as(dim))

            # ---- 主位置一致性 ----
            loc = loc_pred[b][mask]  # [N, 3]
            mean_loc = loc.mean(0, keepdim=True)
            loss_loc = F.mse_loss(loc, mean_loc.expand_as(loc))

            loss_total += (loss_pmt + loss_mad + loss_dim + loss_loc)

    # 归一化
    return 0.2 * (loss_total / bs)


def constraint_loss(xyz, log_pmt_pred, mad_pred, dim_pred, nor_pred, loc_pred,
                    pmt_gt, mad_gt, dim_gt, nor_gt, loc_gt, affil_idx, eps=1e-6):
    """
    计算损失，包含一般损失和几何损失

    点坐标
    :param xyz: [bs, point, 3]

    预测数据
    :param log_pmt_pred: [bs, point, 5]
    :param mad_pred: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param nor_pred: [bs, point, 3]
    :param loc_pred: [bs, point, 3]

    标签数据
    :param pmt_gt: [bs, point] (int, index)
    :param mad_gt: [bs, point, 3]
    :param dim_gt: [bs, point]
    :param nor_gt: [bs, point, 3]
    :param loc_gt: [bs, point, 3]
    :param affil_idx: [bs, point]

    :param eps: 防止除 0 的调整实数
    """

    # 基元类型
    pmt_nll = F.nll_loss(log_pmt_pred.view(-1, 5), pmt_gt.view(-1))

    # # 主方向的长度不应接近 0
    # mad_pred, loss_mad_min_len = safe_normalize(mad_pred)
    #
    # # 法线长度不应接近 0
    # nor_pred, loss_nor_min_len = safe_normalize(nor_pred)

    # 主方向和法线的长度应接近1，注意法线所有基元类型都有，但是主方向只有平面(0)、圆柱(1)、圆锥(2)有
    # unit_len_mad = unit_len_loss_with_pmt_considered(mad_pred, pmt_gt, (0, 1, 2))
    # unit_len_nor = unit_len_loss(nor_pred)

    # 长度归一化
    mad_pred = mad_pred / (mad_pred.norm(dim=-1, keepdim=True) + eps)
    nor_pred = nor_pred / (nor_pred.norm(dim=-1, keepdim=True) + eps)

    # 主方向先进行方向的归一化
    # mad_pred = canonicalize_vectors_hard(mad_pred)

    # 主方向损失
    mad_mse = mse_loss_with_pmt_considered(mad_pred, mad_gt, pmt_gt, (2, ))

    # 主尺寸损失
    dim_mse = mse_loss_with_pmt_considered(dim_pred, dim_gt, pmt_gt, (1, 2, 3))

    # 法线损失
    nor_mse = mse_loss_with_pmt_considered(nor_pred, nor_gt, pmt_gt, (0, 1, 2, 3, 4))

    # 主位置损失
    loc_mse = mse_loss_with_pmt_considered(loc_pred, loc_gt, pmt_gt, (2, ))  # 测试这个损失很大，可能造成训练不稳定，故注释

    # 平面几何损失
    loss_plane = geom_loss_plane(xyz, mad_pred, nor_pred, loc_pred, pmt_gt)

    # 圆柱几何损失
    loss_cylinder = geom_loss_cylinder(xyz, mad_pred, nor_pred, dim_pred, loc_pred, pmt_gt)

    # 圆锥几何损失
    loss_cone = geom_loss_cone(xyz, mad_pred, dim_pred, loc_pred, pmt_gt)

    # 球几何损失
    loss_sphere = geom_loss_sphere(xyz, dim_pred, nor_pred, loc_pred, pmt_gt)

    # 实例一致性损失
    # loss_consistent = instance_consistency_loss(log_pmt_pred, mad_pred, dim_pred, loc_pred, affil_idx)

    # 总损失
    loss_all = pmt_nll + mad_mse + dim_mse + nor_mse + loc_mse + loss_plane + loss_cylinder + loss_cone + loss_sphere

    loss_dict = {
        'all': value_item(loss_all),
        'pmt_nll': value_item(pmt_nll),
        'mad_mse': value_item(mad_mse),
        'dim_mse': value_item(dim_mse),
        'nor_mse': value_item(nor_mse),
        'loc_mse': value_item(loc_mse),
        'loss_plane': value_item(loss_plane),
        'loss_cylinder': value_item(loss_cylinder),
        'loss_cone': value_item(loss_cone),
        'loss_sphere': value_item(loss_sphere),
        # 'loss_mad_unit_len': value_item(unit_len_mad),
        # 'loss_nor_unit_len': value_item(unit_len_nor),
        # 'loss_consistent': value_item(loss_consistent),
    }

    if loss_all.isnan().item() or loss_all.item() >= 20:
        print(loss_dict)

    return loss_all, loss_dict


def canonicalize_vectors_hard(v, eps=1e-6):
    """
    归一化并进行方向标准化的向量处理函数。
    输入 v: [bs, point, 3]
    """

    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    z_zero = (z.abs() < eps)
    y_zero = (y.abs() < eps)
    x_zero = (x.abs() < eps)

    flip_mask = (
        (z < 0) |
        (z_zero & (y < 0)) |
        (z_zero & y_zero & (x < 0))
    )

    flip_mask = flip_mask.unsqueeze(-1)  # [bs, point, 1]
    v = torch.where(flip_mask, -v, v)

    return v


def safe_normalize(v, eps=1e-6, min_norm=0.05):
    """
    v: [bs, point, 3]
    防止向量长度过短
    """
    norm = v.norm(dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    v_normalized = v / norm

    # 只惩罚过短向量，防止不稳定
    length_loss = torch.relu(min_norm - norm).mean()
    return v_normalized, length_loss


def value_item(atensor):
    if isinstance(atensor, float):
        return atensor
    else:
        return atensor.item()


def test():
    pass


if __name__ == '__main__':
    test()


