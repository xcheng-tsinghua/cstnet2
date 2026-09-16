"""
用于包装约束提取器，给出统一表达形式
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import utils, attn_3dgcn, point_net, point_net2


LOC_INPUT = "backbone_xyz_v1"


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


class CstPredWrapper(nn.Module):
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
                 n_prim_type=5,
                 ):
        super().__init__()

        self.loc_input = LOC_INPUT
        self.mad_input = "backbone_xyz_v1"
        self.dim_input = "backbone_xyz_v1"
        self.embedding = get_embedding_model(embedding_model_name, channel_coord, channel_fea, channel_mid)
        self.emb_head = utils.MLP(1, (channel_mid, math.ceil((channel_out*channel_mid)**0.5), channel_out))
        self.cls_head = utils.MLP(1, (channel_mid, math.ceil((n_prim_type*channel_mid)**0.5), n_prim_type))
        attr_mid = math.ceil((3 * channel_mid) ** 0.5)
        dim_mid = math.ceil(channel_mid ** 0.5)
        self.mad_head = utils.MLP(1, (channel_mid + channel_coord, attr_mid, 3), dropout=0.0)
        self.dim_head = utils.MLP(1, (channel_mid + channel_coord, dim_mid, 1), dropout=0.0)
        self.loc_head = utils.MLP(1, (channel_mid + channel_coord, attr_mid, 3), dropout=0.0)

    def build_neighborhood(self, xyz, feature_k=None):
        if not isinstance(self.embedding, attn_3dgcn.Attn3DGcnPointEmbedding):
            return None
        count = max(self.embedding.n_neighbor + 1, self.embedding.attn_k)
        if feature_k is not None:
            count = max(count, max(2, int(feature_k)) + 1)
        return attn_3dgcn.build_neighborhood(xyz, count)

    def forward(self, xyz, fea=None, neighborhood=None):
        """
        xyz: [bs, N, 3]
        fea: [bs, N, channel_fea]
        return: dict containing all Stage 1 semantic and geometry predictions
        """
        xyz = xyz.permute(0, 2, 1)
        if fea is not None:
            fea = fea.permute(0, 2, 1)

        # 提取逐点特征
        if neighborhood is None:
            embedding = self.embedding(xyz, fea)
        else:
            embedding = self.embedding(xyz, fea, neighborhood=neighborhood)

        # 提取聚类特征和基元类型预测特征
        pnt_fea = self.emb_head(embedding)  # -> [bs, fea, n]
        pmt_fea = self.cls_head(embedding)  # -> [bs, fea, n]

        # 将输出的 embedding 进行 L2 正则化，加速聚类
        pnt_fea_l2norm = F.normalize(pnt_fea, dim=1)  # -> [bs, fea, n]

        # 进行 log_softmax 处理，便于后续使用 nll_loss
        pmt_log_softmax = F.log_softmax(pmt_fea, dim=1)  # -> [bs, fea, n]

        # -> [bs, n, fea]
        pnt_fea_l2norm, pmt_log_softmax = pnt_fea_l2norm.permute(0, 2, 1), pmt_log_softmax.permute(0, 2, 1)
        attribute_input = torch.cat([embedding, xyz], dim=1)
        mad_pred = F.normalize(self.mad_head(attribute_input), dim=1, eps=1e-6).permute(0, 2, 1)
        dim_pred = F.softplus(self.dim_head(attribute_input)).squeeze(1)
        loc_pred = self.loc_head(attribute_input).permute(0, 2, 1)

        return {
            "embedding": pnt_fea_l2norm,
            "log_pmt": pmt_log_softmax,
            "mad": mad_pred,
            "dim": dim_pred,
            "loc": loc_pred,
        }

    def set_train_phase(self, phase: str):
        """Configure the exact trainable parameter set for Stage 1 phases."""
        if phase not in ("semantic", "geometry", "joint"):
            raise ValueError(f"unsupported Stage 1 train phase: {phase}")

        for param in self.parameters():
            param.requires_grad_(False)

        if phase == "semantic":
            trainable_prefixes = ["embedding", "emb_head", "cls_head"]
        elif phase == "geometry":
            trainable_prefixes = [
                "mad_head", "dim_head", "loc_head"
            ]
        else:
            trainable_prefixes = [
                "embedding", "emb_head", "cls_head",
                "mad_head", "dim_head", "loc_head",
            ]

        for name, param in self.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in trainable_prefixes):
                param.requires_grad_(True)

        return trainable_prefixes

    def apply_train_phase_mode(self):
        """Keep fully frozen submodules, especially BatchNorm, in eval mode."""
        for module in self.modules():
            parameters = list(module.parameters(recurse=True))
            if parameters and not any(param.requires_grad for param in parameters):
                module.eval()




