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

    # 只对有效类型计算 loss
    loss = F.mse_loss(attr_pred[mask], attr_gt[mask])
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

    xyz = xyz[mask]  # [n_item, 3]
    mad_pred = mad_pred[mask]  # [n_item, 3]
    nor_pred = nor_pred[mask]  # [n_item, 3]
    loc_pred = loc_pred[mask]  # [n_item, 3]

    # 点到垂足的向量与主方向垂直，内积要接近 0
    dot_product = torch.einsum('ij, ij -> i', (loc_pred - xyz), mad_pred).abs().mean()

    # 原点到垂足的向量与主方向平行
    parallel1 = (torch.norm(loc_pred, dim=1) - torch.einsum('ij, ij -> i', loc_pred, mad_pred).abs()).abs().mean()

    # 主方向和法线共线
    parallel2 = (1.0 - torch.einsum('ij, ij -> i', mad_pred, nor_pred).abs()).abs().mean()

    return dot_product + 0.5 * parallel1 + 0.5 * parallel2


def geom_loss_cylinder(xyz, mad_pred, dim_pred, loc_pred, pmt_gt):
    """
    :param xyz: [bs, point, 3]
    :param mad_pred: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部圆柱类型的点
    mask = (pmt_gt == 1)  # [bs, point]

    xyz = xyz[mask]  # [n_item, 3]
    mad_pred = mad_pred[mask]  # [n_item, 3]
    dim_pred = dim_pred[mask]  # [n_item, 3]
    loc_pred = loc_pred[mask]  # [n_item, 3]

    # 半径与预测主尺寸相等
    radius = torch.cross(xyz - loc_pred, mad_pred, dim=1)
    radius = radius.norm(dim=1)
    radius = (radius - dim_pred).abs().mean()

    # 原点到垂足的向量与主方向垂直
    dot_product = torch.einsum('ij, ij -> i', loc_pred, mad_pred).abs().mean()

    return radius + dot_product


def geom_loss_cone(xyz, mad_pred, dim_pred, loc_pred, pmt_gt):
    """
    :param xyz: [bs, point, 3]
    :param mad_pred: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部圆锥类型的点
    mask = (pmt_gt == 2)  # [bs, point]

    xyz = xyz[mask]  # [n_item, 3]
    mad_pred = mad_pred[mask]  # [n_item, 3]
    dim_pred = dim_pred[mask]  # [n_item, ]
    loc_pred = loc_pred[mask]  # [n_item, 3]

    # 从锥角到圆锥面上的点构成的向量与主方向之间的夹角等于主尺寸
    apex_to_xyz = xyz - loc_pred
    dot1 = torch.einsum('ij, ij -> i', mad_pred, apex_to_xyz)
    dot2 = mad_pred.norm(dim=1) * apex_to_xyz.norm(dim=1) * torch.cos(dim_pred)
    semi_angle = (dot1 - dot2).abs().mean()

    return semi_angle


def geom_loss_sphere(xyz, dim_pred, loc_pred, pmt_gt):
    """
    :param xyz: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部球类型的点
    mask = (pmt_gt == 3)  # [bs, point]

    xyz = xyz[mask]  # [n_item, 3]
    dim_pred = dim_pred[mask]  # [n_item, ]
    loc_pred = loc_pred[mask]  # [n_item, 3]

    # 球面上的点到主位置的距离等于主尺寸
    center_to_xyz = xyz - loc_pred
    radius = (center_to_xyz.norm(dim=1) - dim_pred).abs().mean()

    return radius


def constraint_loss(xyz, log_pmt_pred, mad_pred, dim_pred, nor_pred, loc_pred,
                    pmt_gt, mad_gt, dim_gt, nor_gt, loc_gt, affil_idx):
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
    """

    # 基元类型
    pmt_nll = F.nll_loss(log_pmt_pred.view(-1, 5), pmt_gt.view(-1))

    # 主方向损失
    mad_mse = mse_loss_with_pmt_considered(mad_pred, mad_gt, pmt_gt, (0, 1, 2))

    # 主尺寸损失
    dim_mse = mse_loss_with_pmt_considered(dim_pred, dim_gt, pmt_gt, (1, 2, 3))

    # 法线损失
    nor_mse = mse_loss_with_pmt_considered(nor_pred, nor_gt, pmt_gt, (0, 1, 2, 3, 4))

    # 主位置损失
    loc_mse = mse_loss_with_pmt_considered(loc_pred, loc_gt, pmt_gt, (0, 1, 2, 3))

    # 平面几何损失
    loss_plane = geom_loss_plane(xyz, mad_pred, nor_pred, loc_pred, pmt_gt)

    # 圆柱几何损失
    loss_cylinder = geom_loss_cylinder(xyz, mad_pred, dim_pred, loc_pred, pmt_gt)

    # 圆锥几何损失
    loss_cone = geom_loss_cone(xyz, mad_pred, dim_pred, loc_pred, pmt_gt)

    # 球几何损失
    loss_sphere = geom_loss_sphere(xyz, dim_pred, loc_pred, pmt_gt)

    loss_all = pmt_nll + mad_mse + dim_mse + nor_mse + loc_mse + loss_plane + loss_cylinder + loss_cone + loss_sphere

    return loss_all


def test():
    pass


if __name__ == '__main__':
    test()


