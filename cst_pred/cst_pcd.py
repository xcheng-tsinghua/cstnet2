import torch.nn as nn
import torch
import torch.nn.functional as F

from models import utils


class DownSample(nn.Module):
    def __init__(self, n_center, n_near, channel_in, channel_out):
        super().__init__()

        self.n_center = n_center
        self.n_near = n_near

        channel_mid = int(((channel_in + 3) * channel_out) ** 0.5)
        self.mlp = utils.MLP(2, (channel_in + 3, channel_mid, channel_out), final_proc=True)

    def forward(self, xyz, fea):
        """
        xyz: [bs, n_point, 3]
        fea: [bs, n_point, f]
        """
        idx_surfknn_all = utils.knn(xyz, self.n_near)

        # 采样后的中心点索引 [bs, n_center]
        fps_idx = utils.fps(xyz, self.n_center)

        # 采样后的中心点坐标 [bs, n_center, 3]
        center_xyz = utils.index_points(xyz, fps_idx)

        # 采样后的周围点索引 [bs, n_center, n_near]
        group_idx = utils.index_points(idx_surfknn_all, fps_idx)

        # 采样后的周围点坐标 [bs, n_center, n_near, 3]
        group_xyz = utils.index_points(xyz, group_idx)

        # 中心点到周围点的向量 [bs, n_center, n_near, 3]
        xyz_relative = group_xyz - center_xyz.unsqueeze(2)

        # 采样后的周围点坐标 [bs, n_center, n_near, channel]
        group_fea = utils.index_points(fea, group_idx)

        # 采样后的周围点坐标 [bs, channel + 3, n_center, n_near]
        group_fea = torch.cat([group_fea, xyz_relative], dim=-1).permute(0, 3, 1, 2)

        # 更新特征 [bs, channel, n_center, n_near]
        new_fea = self.mlp(group_fea)

        # [bs, n_center, channel]
        new_fea = new_fea.max(3)[0].permute(0, 2, 1)

        return center_xyz, new_fea


class UpSample(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = utils.MLP(1, mlp)

    def forward(self, xyz1, xyz2, fea1, fea2):
        """
        xyz1: [bs, n_point1, 3]
        xyz2: [bs, n_point2, 3]
        fea1: [bs, n_point1, channel1]
        fea2: [bs, n_point2, channel2]
        """
        bs, n_point, _ = xyz1.shape

        dists = utils.square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)

        weight = dist_recip / norm
        interpolated_fea = torch.sum(utils.index_points(fea2, idx) * weight.view(bs, n_point, 3, 1), dim=2)

        # skip link concatenation
        new_fea = torch.cat([fea1, interpolated_fea], dim=-1)
        new_fea = self.mlp(new_fea.permute(0, 2, 1)).permute(0, 2, 1)

        return new_fea


class CstPcd(nn.Module):
    def __init__(self, n_points_all, sample_rate=0.9):
        super().__init__()
        print('constraint prediction original version')

        self.dn1 = DownSample(int(n_points_all * sample_rate), 50, 3, 64)
        self.dn2 = DownSample(int(n_points_all * sample_rate ** 2), 40, 64, 128)
        self.dn3 = DownSample(int(n_points_all * sample_rate ** 3), 30, 128, 256)

        self.up3 = UpSample((256 + 128, 256, 128))
        self.up2 = UpSample((128 + 64, 128, 64))
        self.up1 = UpSample((64 + 6, 64, 32))

        self.mlp_pmt = utils.MLP(1, (32, 16, 5))  # 5 类基元
        self.mlp_mad = utils.MLP(1, (32, 16, 3))  # 主方向 3 个坐标分量
        self.mlp_dim = utils.MLP(1, (32, 16, 1))  # 主尺寸 1 个实数
        self.mlp_nor = utils.MLP(1, (32, 16, 3))  # 法线 3 个坐标分量
        self.mlp_loc = utils.MLP(1, (32, 16, 3))  # 主位置 3 个坐标分量

    def forward(self, xyz):
        """
        xyz: [bs, n_point, 3]
        """
        l1_xyz, l1_fea = self.dn1(xyz, xyz)
        l2_xyz, l2_fea = self.dn2(l1_xyz, l1_fea)
        l3_xyz, l3_fea = self.dn3(l2_xyz, l2_fea)

        l2_fea = self.up3(l2_xyz, l3_xyz, l2_fea, l3_fea)
        l1_fea = self.up2(l1_xyz, l2_xyz, l1_fea, l2_fea)
        l0_fea = self.up1(xyz, l1_xyz, torch.cat([xyz, xyz], 2), l1_fea).permute(0, 2, 1)

        # FC layers
        pmt = self.mlp_pmt(l0_fea).permute(0, 2, 1)  # [bs, n_points_all, 5]
        log_pmt = F.log_softmax(pmt, dim=2)  # 分类，使用 log_softmax

        mad = self.mlp_mad(l0_fea).permute(0, 2, 1)  # [bs, n_points_all, 3]
        dim = self.mlp_dim(l0_fea).squeeze()  # [bs, n_points_all]
        nor = self.mlp_nor(l0_fea).permute(0, 2, 1)  # [bs, n_points_all, 3]
        loc = self.mlp_loc(l0_fea).permute(0, 2, 1)  # [bs, n_points_all, 4]

        return log_pmt, mad, dim, nor, loc


class CstPcdSimplify(nn.Module):
    def __init__(self, n_points_all, sample_rate=0.9):
        super().__init__()
        print('simplified constraint prediction from point cloud')

        self.dn1 = DownSample(int(n_points_all * sample_rate), 50, 3, 64)
        self.dn2 = DownSample(int(n_points_all * sample_rate ** 2), 40, 64, 128)

        self.up2 = UpSample((128 + 64, 128, 64))
        self.up1 = UpSample((64 + 6, 64, 32))

        self.mlp_pmt = utils.MLP(1, (32, 16, 5))  # 5 类基元
        self.mlp_mad = utils.MLP(1, (32, 16, 3))  # 主方向 3 个坐标分量
        self.mlp_dim = utils.MLP(1, (32, 16, 1))  # 主尺寸 1 个实数
        self.mlp_nor = utils.MLP(1, (32, 16, 3))  # 法线 3 个坐标分量
        self.mlp_loc = utils.MLP(1, (32, 16, 3))  # 主位置 3 个坐标分量

    def forward(self, xyz):
        """
        xyz: [bs, n_point, 3]
        """
        l1_xyz, l1_fea = self.dn1(xyz, xyz)
        l2_xyz, l2_fea = self.dn2(l1_xyz, l1_fea)

        l1_fea = self.up2(l1_xyz, l2_xyz, l1_fea, l2_fea)
        l0_fea = self.up1(xyz, l1_xyz, torch.cat([xyz, xyz], 2), l1_fea).permute(0, 2, 1)

        # FC layers
        pmt = self.mlp_pmt(l0_fea).permute(0, 2, 1)  # [bs, n_points_all, 5]
        log_pmt = F.log_softmax(pmt, dim=2)  # 分类，使用 log_softmax

        mad = self.mlp_mad(l0_fea).permute(0, 2, 1)  # [bs, n_points_all, 3]
        dim = self.mlp_dim(l0_fea).squeeze()  # [bs, n_points_all]
        nor = self.mlp_nor(l0_fea).permute(0, 2, 1)  # [bs, n_points_all, 3]
        loc = self.mlp_loc(l0_fea).permute(0, 2, 1)  # [bs, n_points_all, 4]

        return log_pmt, mad, dim, nor, loc


if __name__ == '__main__':
    test_tensor = torch.rand(16, 2000, 3).cuda()
    anet = CstPcdSimplify(2000).cuda()

    _log_pmt, _mad, _dim, _nor, _loc = anet(test_tensor)
    print(_log_pmt.size(), _mad.size(), _dim.size(), _nor.size(), _loc.size())



