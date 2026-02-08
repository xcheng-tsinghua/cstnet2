"""
用于包装约束提取器，给出统一表达形式
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from modules import utils, attn_3dgcn, point_net, point_net2


def get_embedding_model(name, channel_coord=3, channel_fea=0, channel_mid=128,):
    """
    根据模型名获取对应特征提取模块
    Args:
        name:
        channel_coord:
        channel_fea:
        channel_mid:

    Returns:

    """
    if name == 'attn_3dgcn':
        print('-> create attn_3dgcn point embedding')
        embedding_model = attn_3dgcn.Attn3DGcnPointEmbedding(channel_coord, channel_fea, channel_mid)

    elif name == 'pointnet':
        print('-> create pointnet point embedding')
        embedding_model = point_net.PointNetPointEmbedding(channel_coord, channel_fea, channel_mid)

    elif name == 'pointnet2':
        print('-> create pointnet2 point embedding')
        embedding_model = point_net2.PointNet2PointEmbedding(channel_coord, channel_fea, channel_mid)

    else:
        raise ValueError(f'not support {name}')

    return embedding_model


class CstModelWrapper(nn.Module):
    """
    多层 3DGCN 特征编码器
    输入: xyz [bs, 3, N]
    输出: fea [bs, channel_out, N]
    """
    def __init__(self,
                 embedding_model_name: str,
                 channel_coord=3,
                 channel_fea=0,
                 channel_mid=128,
                 channel_out=32,
                 n_prim_type=5
                 ):
        super().__init__()

        self.embedding = get_embedding_model(embedding_model_name, channel_coord, channel_fea, channel_mid)
        self.emb_head = utils.MLP(1, (channel_mid, math.ceil((channel_out*channel_mid)**0.5), channel_out))
        self.cls_head = utils.MLP(1, (channel_mid, math.ceil((n_prim_type*channel_mid)**0.5), n_prim_type))

    def forward(self, xyz, fea=None):
        """
        xyz: [bs, N, 3]
        fea: [bs, N, channel_fea]
        return: [bs, N, channel_out], [bs, N, channel_out]
        """
        xyz, fea = xyz.permute(0, 2, 1), fea.permute(0, 2, 1)

        # 提取逐点特征
        embedding = self.embedding(xyz, fea)  # -> [bs, fea, n]

        # 提取聚类特征和基元类型预测特征
        pnt_fea = self.emb_head(embedding)  # -> [bs, fea, n]
        pmt_fea = self.cls_head(embedding)  # -> [bs, fea, n]

        # 将输出的 embedding 进行 L2 正则化，加速聚类
        pnt_fea_l2norm = F.normalize(pnt_fea, dim=1)  # -> [bs, fea, n]

        # 进行 log_softmax 处理，便于后续使用 nll_loss
        pmt_log_softmax = F.log_softmax(pmt_fea, dim=1)  # -> [bs, fea, n]

        # -> [bs, n, fea]
        pnt_fea_l2norm, pmt_log_softmax = pnt_fea_l2norm.permute(0, 2, 1), pmt_log_softmax.permute(0, 2, 1)
        return pnt_fea_l2norm, pmt_log_softmax




