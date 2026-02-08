import torch.nn as nn
import torch
import torch.nn.functional as F

from modules import point_net2
from modules import utils


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
    def __init__(self):
        super().__init__()
        print('constraint prediction original version')

        self.point_fea_encoder = pointnet2.PointNet2PointFeaEncoder()  # 获取逐点特征
        self.point_pmt_mlp = utils.MLP(1, (128, 64, 5))  # 5类基元

    def forward(self, xyz):
        """
        xyz: [bs, 3, n_points]

        return: pmt_log: [bs, n_points, 5], raw_fea: [bs, 128, n_points]
        """
        raw_fea = self.point_fea_encoder(xyz)  # [bs, fea, n_points]

        pmt_fea = self.point_pmt_mlp(raw_fea)  # [bs, fea, n_points]
        pmt_log = F.log_softmax(pmt_fea, 1)
        pmt_log = pmt_log.transpose(1, 2)  # [bs, n_points, 5]

        return pmt_log, raw_fea


if __name__ == '__main__':
    test_tensor = torch.rand(16, 2000, 3).cuda()
    anet = CstPcd().cuda()

    _log_pmt, _mad, _dim, _nor, _loc = anet(test_tensor)
    print(_log_pmt.size(), _mad.size(), _dim.size(), _nor.size(), _loc.size())



