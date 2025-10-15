import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models import utils
from data_utils.vis import vis_3d_points_knn

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# class QuarticSSA(nn.Module):
#     def __init__(self, n_center, n_near, n_stepk, n_primitive, channel_in, channel_out):
#         """
#         n_center: sampled points in this layer
#         n_near: neighborhoods of center points
#         n_stepk: each step neighbor in SurfaceKNN
#         mlp: channels of MLP
#
#         """
#         super().__init__()
#         self.n_center = n_center
#         self.n_near = n_near
#         self.n_stepk = n_stepk
#
#         self.attention_points_mad = utils.PointAttention(channel_in, channel_in + 3, channel_out)
#         self.attention_points_adj = utils.PointAttention(channel_in, channel_in + 4, channel_out)
#         self.attention_points_pt = utils.PointAttention(channel_in, channel_in + n_primitive * 2, channel_out)
#         self.attention_points_cst = utils.PointAttention(channel_in, channel_in + 3, channel_out)
#
#         self.mlp_mad = utils.MLP(0, (channel_in + 3, channel_out, channel_out))
#         self.mlp_adj = utils.MLP(0, (channel_in + 4, channel_out, channel_out))
#         self.mlp_pt = utils.MLP(0, (channel_in + n_primitive * 2, channel_out, channel_out))
#         self.mlp_cst = utils.MLP(0, (channel_in + 3, channel_out, channel_out))
#
#     def forward(self, xyz, mad, adj, pt, mad_fea, adj_fea, pt_fea, cst_fea):
#         """
#         xyz: [bs, n_point, 3]
#         mad: [bs, n_point, 3]
#         adj: [bs, n_point, 2] one-hot
#         pt: [bs, n_point, 4] one-hot
#
#         mad_fea: [bs, n_point, n_channel]
#         adj_fea: [bs, n_point, n_channel]
#         pt_fea: [bs, n_point, n_channel]
#         cst_fea: [bs, n_point, n_channel]
#
#         """
#
#         if self.n_center is None:  # group all point into a center
#
#             g_mad = mad
#             g_adj = adj.repeat(1, 1, 2)
#             g_pt = pt.repeat(1, 1, 2)
#             g_xyz = xyz
#
#             g_mad_fea = mad_fea
#             g_adj_fea = adj_fea
#             g_pt_fea = pt_fea
#             g_cst_fea = cst_fea
#
#             g_mad_fea = torch.cat([g_mad_fea, g_mad], dim=-1).max(1)[0]
#             g_adj_fea = torch.cat([g_adj_fea, g_adj], dim=-1).max(1)[0]
#             g_pt_fea = torch.cat([g_pt_fea, g_pt], dim=-1).max(1)[0]
#             g_cst_fea = torch.cat([g_cst_fea, g_xyz], dim=-1).max(1)[0]
#
#             g_mad_fea = self.mlp_mad(g_mad_fea)
#             g_adj_fea = self.mlp_adj(g_adj_fea)
#             g_pt_fea = self.mlp_pt(g_pt_fea)
#             g_cst_fea = self.mlp_cst(g_cst_fea)
#
#             return g_mad_fea, g_adj_fea, g_pt_fea, g_cst_fea
#
#         else:
#             idx_surfknn_all = utils.surface_knn(xyz, self.n_near, self.n_stepk)  # [bs, n_point, n_near]
#             fps_idx = utils.fps(xyz, self.n_center)  # [bs, n_center]
#
#             center_xyz = utils.index_points(xyz, fps_idx)  # [bs, n_center, 3]
#             center_mad = utils.index_points(mad, fps_idx)
#             center_adj = utils.index_points(adj, fps_idx)
#             center_pt = utils.index_points(pt, fps_idx)
#
#             center_mad_fea = utils.index_points(mad_fea, fps_idx)  # [bs, n_center, f]
#             center_adj_fea = utils.index_points(adj_fea, fps_idx)
#             center_pt_fea = utils.index_points(pt_fea, fps_idx)
#             center_cst_fea = utils.index_points(cst_fea, fps_idx)
#
#             idx = utils.index_points(idx_surfknn_all, fps_idx)  # [bs, n_center, n_near]
#
#             g_xyz = utils.index_points(xyz, idx)  # [bs, n_center, n_near, 3]
#             g_mad = utils.index_points(mad, idx)
#             g_adj = utils.index_points(adj, idx)
#             g_pt = utils.index_points(pt, idx)
#
#             g_mad_fea = utils.index_points(mad_fea, idx)  # [bs, n_center, n_near, 3]
#             g_adj_fea = utils.index_points(adj_fea, idx)
#             g_pt_fea = utils.index_points(pt_fea, idx)
#             g_cst_fea = utils.index_points(cst_fea, idx)
#
#             mad_relative = g_mad - center_mad.unsqueeze(2)  # [bs, n_center, n_near, 3]
#             adj_cat = torch.cat([g_adj, center_adj.unsqueeze(2).repeat(1, 1, self.n_near, 1)], dim=-1)
#             pt_cat = torch.cat([g_pt, center_pt.unsqueeze(2).repeat(1, 1, self.n_near, 1)], dim=-1)
#             xyz_relative = g_xyz - center_xyz.unsqueeze(2)
#
#             g_mad_fea = torch.cat([g_mad_fea, mad_relative], dim=-1)  # [bs, n_center, n_near, f]
#             g_adj_fea = torch.cat([g_adj_fea, adj_cat], dim=-1)
#             g_pt_fea = torch.cat([g_pt_fea, pt_cat], dim=-1)
#             g_cst_fea = torch.cat([g_cst_fea, xyz_relative], dim=-1)
#
#             g_mad_fea = self.attention_points_mad(center_mad_fea, g_mad_fea)
#             g_adj_fea = self.attention_points_adj(center_adj_fea, g_adj_fea)
#             g_pt_fea = self.attention_points_pt(center_pt_fea, g_pt_fea)
#             g_cst_fea = self.attention_points_cst(center_cst_fea, g_cst_fea)
#
#             return center_xyz, center_mad, center_adj, center_pt, g_mad_fea, g_adj_fea, g_pt_fea, g_cst_fea

class MultiSSA(nn.Module):
    def __init__(self, n_center, n_near, n_stepk, channel_in, channel_out):
        """
        n_center: sampled points in this layer
        n_near: neighborhoods of center points
        n_stepk: each step neighbor in SurfaceKNN
        mlp: channels of MLP

        """
        super().__init__()
        self.n_center = n_center
        self.n_near = n_near
        self.n_stepk = n_stepk

        self.attention_points_xyz = utils.PointAttention2(channel_in, channel_in)
        self.attention_points_pmt = utils.PointAttention2(channel_in, channel_in)
        self.attention_points_mad = utils.PointAttention2(channel_in, channel_in)
        self.attention_points_dim = utils.PointAttention2(channel_in, channel_in)
        self.attention_points_nor = utils.PointAttention2(channel_in, channel_in)
        self.attention_points_loc = utils.PointAttention2(channel_in, channel_in)
        self.attention_points_fea = utils.PointAttention2(channel_in, channel_in)

        self.mlp_xyz = utils.MLP(0, (channel_in, channel_out, channel_out))
        self.mlp_pmt = utils.MLP(0, (channel_in, channel_out, channel_out))
        self.mlp_mad = utils.MLP(0, (channel_in, channel_out, channel_out))
        self.mlp_dim = utils.MLP(0, (channel_in, channel_out, channel_out))
        self.mlp_nor = utils.MLP(0, (channel_in, channel_out, channel_out))
        self.mlp_loc = utils.MLP(0, (channel_in, channel_out, channel_out))
        self.mlp_fea = utils.MLP(0, (channel_in, channel_out, channel_out))

    def forward(self, xyz_fea, pmt_fea, mad_fea, dim_fea, nor_fea, loc_fea, fea):
        """
        xyz_fea: [bs, n_point, n_channel]
        pmt_fea: [bs, n_point, n_channel]
        mad_fea: [bs, n_point, n_channel]
        dim_fea: [bs, n_point, n_channel]
        nor_fea: [bs, n_point, n_channel]
        loc_fea: [bs, n_point, n_channel]
        fea: [bs, n_point, n_channel]
        """
        bs, n_point, dim = xyz_fea.shape
        idx_surfknn_all = utils.surface_knn(xyz_fea, self.n_near, self.n_stepk)  # [bs, n_point, n_near]
        # idx_knn_all = utils.knn(xyz_fea, self.n_near)  # [bs, n_point, n_near]
        fps_idx = utils.fps(xyz_fea, self.n_center, dim=dim)  # [bs, n_center]

        center_xyz_fea = utils.index_points(xyz_fea, fps_idx)  # [bs, n_center, f]
        center_pmt_fea = utils.index_points(pmt_fea, fps_idx)
        center_mad_fea = utils.index_points(mad_fea, fps_idx)
        center_dim_fea = utils.index_points(dim_fea, fps_idx)
        center_nor_fea = utils.index_points(nor_fea, fps_idx)
        center_loc_fea = utils.index_points(loc_fea, fps_idx)
        center_fea = utils.index_points(fea, fps_idx)
        # print("xyz_fea.shape: ", center_xyz_fea.shape)
        # print("idx_surfknn_all.shape: ", idx_surfknn_all.shape)
        # print("fps_idx.shape: ", fps_idx.shape)

        idx = utils.index_points(idx_surfknn_all, fps_idx)  # [bs, n_center, n_near]

        # print("idx.shape: ", idx.shape)

        g_xyz = utils.index_points(xyz_fea, idx)  # [bs, n_center, n_near, f]
        g_pmt = utils.index_points(pmt_fea, idx)
        g_mad = utils.index_points(mad_fea, idx)
        g_dim = utils.index_points(dim_fea, idx)
        g_nor = utils.index_points(nor_fea, idx)
        g_loc = utils.index_points(loc_fea, idx)
        g_fea = utils.index_points(fea, idx)

        # print("g_xyz.shape: ", g_xyz.shape)
        # print("center_xyz_fea.shape: ", center_xyz_fea.shape)

        g_xyz_fea = self.attention_points_xyz(center_xyz_fea, g_xyz)
        g_pmt_fea = self.attention_points_pmt(center_pmt_fea, g_pmt)
        g_mad_fea = self.attention_points_mad(center_mad_fea, g_mad)
        g_dim_fea = self.attention_points_dim(center_dim_fea, g_dim)
        g_nor_fea = self.attention_points_nor(center_nor_fea, g_nor)
        g_loc_fea = self.attention_points_loc(center_loc_fea, g_loc)
        g_fea = self.attention_points_fea(center_fea, g_fea)

        g_xyz_fea = self.mlp_xyz(g_xyz_fea)
        g_pmt_fea = self.mlp_pmt(g_pmt_fea)
        g_mad_fea = self.mlp_mad(g_mad_fea)
        g_dim_fea = self.mlp_dim(g_dim_fea)
        g_nor_fea = self.mlp_nor(g_nor_fea)
        g_loc_fea = self.mlp_loc(g_loc_fea)
        g_fea = self.mlp_fea(g_fea)

        return g_xyz_fea, g_pmt_fea, g_mad_fea, g_dim_fea, g_nor_fea, g_loc_fea, g_fea


class CMLP(nn.Module):
    def __init__(self, n_primitive=4, mlp_ex0=(16, 32)):
        """
        n_primitive: classes number of primitives
        mlp_ex0: mlp channels except layer 0
        """
        super().__init__()
        self.cst_mlp = utils.MLP(1, (3 + 3 + 2 + n_primitive,) + mlp_ex0)

    def forward(self, xyz, mad, adj, pt):
        """
        xyz: [bs, n_point, 3]
        mad: [bs, n_point, 3]
        adj: [bs, n_point, 2] one-hot
        pt: [bs, n_point, 4] one-hot
        fea: [bs, n_point, n_channel] one-hot
        """
        xyz_cst = torch.cat([xyz, mad, adj, pt], dim=2).permute(0, 2, 1)
        cst_fea = self.cst_mlp(xyz_cst).permute(0, 2, 1)

        return cst_fea

class CstMLP(nn.Module):
    def __init__(self, channel_in=8 * 6, mlp_ex0=(8 * 3, 8)):
        """
        n_primitive: classes number of primitives
        mlp_ex0: mlp channels except layer 0
        """
        super().__init__()
        self.cst_mlp = utils.MLP(1, (channel_in,) + mlp_ex0)

    def forward(self, xyz_fea, pmt_fea, mad_fea, dim_fea, nor_fea, loc_fea):
        """
        xyz: [bs, n_point, 3]
        mad: [bs, n_point, 3]
        adj: [bs, n_point, 2] one-hot
        pt: [bs, n_point, 4] one-hot
        fea: [bs, n_point, n_channel] one-hot
        """
        xyz_fea = xyz_fea.permute(0, 2, 1)
        pmt_fea = pmt_fea.permute(0, 2, 1)
        mad_fea = mad_fea.permute(0, 2, 1)
        dim_fea = dim_fea.permute(0, 2, 1)
        nor_fea = nor_fea.permute(0, 2, 1)
        loc_fea = loc_fea.permute(0, 2, 1)
        xyz_cst = torch.cat([xyz_fea, pmt_fea, mad_fea, dim_fea, nor_fea, loc_fea], dim=2).permute(0, 2, 1)
        # print("xyz_cst.shape: ", xyz_cst.shape)
        cst_fea = self.cst_mlp(xyz_cst)
        # print("cst_fea.shape: ", cst_fea.shape)

        return cst_fea


class Projector(nn.Module):
    def __init__(self, n_primitive=5, mlp_ex0=(8, 8)):
        """
        Initialize a projector
        :param n_primitive: number of primitives
        :param mlp_ex0: mlp layers
        """
        super().__init__()
        self.mlp_xyz = utils.MLP(1, (3,) + mlp_ex0)
        self.mlp_pmt = utils.MLP(1, (n_primitive,) + mlp_ex0)
        self.mlp_mad = utils.MLP(1, (3,) + mlp_ex0)
        self.mlp_dim = utils.MLP(1, (1,) + mlp_ex0)
        self.mlp_nor = utils.MLP(1, (3,) + mlp_ex0)
        self.mlp_loc = utils.MLP(1, (3,) + mlp_ex0)
        self.cst_mlp = CstMLP()

    def forward(self, xyz, pmt, mad, dim, nor, loc):
        """
        xyz: [bs, n_point, 3]
        pmt: [bs, n_point, 5] one-hot
        mad: [bs, n_point, 3]
        dim: [bs, n_point, 1]
        nor: [bs, n_point, 3]
        loc: [bs, n_point, 3]
        """
        xyz, pmt, mad, dim, nor, loc = (xyz.permute(0, 2, 1), pmt.permute(0, 2, 1), mad.permute(0, 2, 1),
                                        dim.permute(0, 2, 1), nor.permute(0, 2, 1), loc.permute(0, 2, 1))
        xyz_fea = self.mlp_xyz(xyz)
        pmt_fea = self.mlp_pmt(pmt)
        mad_fea = self.mlp_mad(mad)
        dim_fea = self.mlp_dim(dim)
        nor_fea = self.mlp_nor(nor)
        loc_fea = self.mlp_loc(loc)
        print("xyz_fea: ", xyz_fea.shape, "pmt_fea: ", pmt_fea.shape, "mad_fea: ", mad_fea.shape, "dim_fea: ", dim_fea.shape, "nor_fea: ", nor_fea.shape, "loc_fea: ", loc_fea.shape)
        cst_fea = self.cst_mlp(xyz_fea, pmt_fea, mad_fea, dim_fea,nor_fea, loc_fea)

        return (xyz_fea.permute(0, 2, 1), pmt_fea.permute(0, 2, 1), mad_fea.permute(0, 2, 1),
                dim_fea.permute(0, 2, 1), nor_fea.permute(0, 2, 1), loc_fea.permute(0, 2, 1),
                cst_fea.permute(0, 2, 1))


#
#
# class TripleMLP(nn.Module):
#     def __init__(self, n_primitive=4, mlp_ex0=(16, 32)):
#         """
#         n_primitive: classes number of primitives
#         mlp_ex0: mlp channels except layer 0
#         """
#         super().__init__()
#
#         self.mlp_xyz = utils.MLP(1, (3,) + mlp_ex0)
#         self.mlp_pmt = utils.MLP(1, (5,) + mlp_ex0)
#         self.mlp_mad = utils.MLP(1, (3,) + mlp_ex0)
#         self.mlp_dim = utils.MLP(1, (1,) + mlp_ex0)
#         self.mlp_nor = utils.MLP(1, (3,) + mlp_ex0)
#         self.mlp_loc = utils.MLP(1, (3,) + mlp_ex0)
#
#         self.mlp_fea = utils.MLP(1, (mlp_ex0[-1] * 6, int(mlp_ex0[-1] * (6 ** 0.5)), mlp_ex0[-1]))
#
#     def forward(self, xyz, pmt, mad, dim, nor, loc):
#         """
#         xyz: [bs, n_point, 3]
#         pmt: [bs, n_point, 5] one-hot
#         mad: [bs, n_point, 3]
#         dim: [bs, n_point, 1]
#         nor: [bs, n_point, 3]
#         loc: [bs, n_point, 3]
#         """
#         xyz, pmt, mad, dim, nor, loc = (xyz.permute(0, 2, 1), pmt.permute(0, 2, 1), mad.permute(0, 2, 1),
#                                         dim.permute(0,2,1), nor.permute(0, 2, 1), loc.permute(0, 2, 1))
#         xyz_fea = self.mlp_xyz(xyz)
#         pmt_fea = self.mlp_pmt(pmt)
#         mad_fea = self.mlp_mad(mad)
#         dim_fea = self.mlp_dim(dim)
#         nor_fea = self.mlp_nor(nor)
#         loc_fea = self.mlp_loc(loc)
#
#         cst_fea = self.mlp_fea(torch.cat([xyz_fea, pmt_fea, mad_fea, dim_fea, nor_fea, loc_fea], dim=-2))
#         return (xyz_fea.permute(0, 2, 1), pmt_fea.permute(0, 2, 1), mad_fea.permute(0, 2, 1),
#                 dim_fea.permute(0, 2, 1), nor_fea.permute(0, 2, 1), loc_fea.permute(0, 2, 1), cst_fea.permute(0, 2, 1))


class CstNet2S2(nn.Module):
    def __init__(self, n_classes, n_primitive):
        super().__init__()
        print('creating CstNet Stage2 network ...')

        channel_l0 = 8
        channel_l1 = 32
        channel_l2 = 128
        channel_l3 = 256

        self.maxpooling = nn.MaxPool1d(kernel_size=channel_l3)

        self.projector = Projector(n_primitive)
        self.multi_ssa1 = MultiSSA(1024, 32, 10, channel_l0, channel_l1)
        self.fea_attn1 = utils.FeaAttention2(channel_l1, channel_l1)

        self.multi_ssa2 = MultiSSA(512, 32, 10, channel_l1, channel_l2)
        self.fea_attn2 = utils.FeaAttention2(channel_l2, channel_l2)

        self.multi_ssa3 = MultiSSA(256, 32, 10, channel_l2, channel_l3)
        self.fea_attn3 = utils.FeaAttention2(channel_l3, channel_l3)

        # self.c_mlp = CMLP(n_primitive, (16, channel_l0))
        #
        # self.quartic_ssa1 = QuarticSSA(1024, 32, 5, n_primitive, channel_l0, channel_l1)
        # self.fea_attn1 = utils.FeaAttention(channel_l1, channel_l1)
        #
        # self.quartic_ssa2 = QuarticSSA(512, 32, 5, n_primitive, channel_l1, channel_l2)
        # self.fea_attn2 = utils.FeaAttention(channel_l2, channel_l2)
        #
        # self.quartic_ssa3 = QuarticSSA(None, None, None, n_primitive, channel_l2, channel_l3)
        # self.fea_attn3 = utils.FeaAttention(channel_l3, channel_l3)

        self.linear = utils.MLP(0, (channel_l3, channel_l2, n_classes))

    def forward(self, xyz, pmt, mad, dim, nor, loc):
        """
        xyz: [bs, n_point, 3]
        pmt: [bs, n_point, 5] one-hot
        mad: [bs, n_point, 3]
        nor: [bs, n_point, 3]
        loc: [bs, n_point, 3]
        """

        xyz_fea, pmt_fea, mad_fea, dim_fea, nor_fea, loc_fea, cst_fea = self.projector(xyz, pmt, mad, dim, nor, loc)  #-> [bs, n_point, f=32]
        # print("cst_fea.shape: ", cst_fea.shape)

        xyz_fea_l1, pmt_fea_l1, mad_fea_l1, dim_fea_l1, nor_fea_l1, loc_fea_l1, cst_fea_l1 = self.multi_ssa1(xyz_fea, pmt_fea, mad_fea, dim_fea, nor_fea, loc_fea, cst_fea)
        cst_fea_l1 = self.fea_attn1(xyz_fea_l1, pmt_fea_l1, mad_fea_l1, dim_fea_l1, nor_fea_l1, loc_fea_l1, cst_fea_l1)

        xyz_fea_l2, pmt_fea_l2, mad_fea_l2, dim_fea_l2, nor_fea_l2, loc_fea_l2, cst_fea_l2 = self.multi_ssa2(xyz_fea_l1, pmt_fea_l1, mad_fea_l1, dim_fea_l1, nor_fea_l1, loc_fea_l1, cst_fea_l1)
        cst_fea_l2 = self.fea_attn2(xyz_fea_l2, pmt_fea_l2, mad_fea_l2, dim_fea_l2, nor_fea_l2, loc_fea_l2, cst_fea_l2)

        xyz_fea_l3, pmt_fea_l3, mad_fea_l3, dim_fea_l3, nor_fea_l3, loc_fea_l3, cst_fea_l3 = self.multi_ssa3(xyz_fea_l2, pmt_fea_l2, mad_fea_l2, dim_fea_l2, nor_fea_l2, loc_fea_l2, cst_fea_l2)
        cst_fea_l3 = self.fea_attn3(xyz_fea_l3, pmt_fea_l3, mad_fea_l3, dim_fea_l3, nor_fea_l3, loc_fea_l3, cst_fea_l3)

        cst_fea_l3_pooled = self.maxpooling(cst_fea_l3.permute(0, 2, 1)).permute(0, 2, 1)
        cls = self.linear(cst_fea_l3_pooled)[:, 0, :]
        cls = F.log_softmax(cls, -1)
        return cls


if __name__ == '__main__':
    xyz_tensor = torch.rand(2, 2048, 3).cuda()
    pmt_tensor = torch.rand(2, 2048, 5).cuda()
    mad_tensor = torch.rand(2, 2048, 3).cuda()
    dim_tensor = torch.rand(2, 2048, 1).cuda()
    nor_tensor = torch.rand(2, 2048, 3).cuda()
    loc_tensor = torch.rand(2, 2048, 3).cuda()

    cstnet = CstNet2S2(5, 5).cuda()
    cls = cstnet(xyz_tensor, pmt_tensor, mad_tensor, dim_tensor, nor_tensor, loc_tensor)
    print(cls.shape)
    # xyz = torch.rand(2, 2048, 8).to("cuda")
    # mlp_xyz = utils.MLP(0, (8, 1024, 32)).to("cuda")
    # xyz_fea = mlp_xyz(xyz)
    # print(xyz_fea.shape)



    # tri_mlp = Projector(5).to("cuda")
    # xyz_fea, pmt_fea, mad_fea, dim_fea, nor_fea, loc_fea = tri_mlp(xyz_tensor, pmt_tensor, mad_tensor, dim_tensor, nor_tensor, loc_tensor)
    # print(xyz_fea.shape, pmt_fea.shape, mad_fea.shape, dim_fea.shape, nor_fea.shape, loc_fea.shape)

    # vis_3d_points_knn(xyz_tensor, 0, utils.surface_knn(xyz_tensor, 100, 10))

