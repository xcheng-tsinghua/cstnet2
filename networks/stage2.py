from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from functional.constraints import CONSTRAINT_DIM, split_constraint_tensor
from networks import utils


class PointwiseMLP(nn.Module):
    """Conv1d-based MLP that operates pointwise over a point cloud."""

    def __init__(self, channels: tuple[int, ...], activate_last: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=False))
            if i < len(channels) - 2 or activate_last:
                layers.append(nn.LayerNorm(channels[i + 1]))
                layers.append(nn.SiLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, C] -> [B, N, C_out]
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ConstraintStreamInitializer(nn.Module):
    """
    Initialize six independent feature streams from xyz and constraint tensor.

    Five component streams: primitive_type, direction, dimension, continuity, location.
    One main constraint stream: initialized from xyz only.

    Output: Six feature tensors [B, N, feature_dim]
    """

    def __init__(self, feature_dim: int = 96):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                "primitive_type": PointwiseMLP((3 + 5, feature_dim, feature_dim)),
                "direction": PointwiseMLP((3 + 3, feature_dim, feature_dim)),
                "dimension": PointwiseMLP((3 + 1, feature_dim, feature_dim)),
                "continuity": PointwiseMLP((3 + 3, feature_dim, feature_dim)),
                "location": PointwiseMLP((3 + 3, feature_dim, feature_dim)),
                "xyz": PointwiseMLP((3, feature_dim, feature_dim)),
            }
        )

    def forward(self, xyz: torch.Tensor, constraints: torch.Tensor) -> Dict[str, torch.Tensor]:
        # xyz: [B, N, 3]
        # constraints: [B, N, CONSTRAINT_DIM]
        # Output: dict with keys ["primitive_type", "direction", "dimension", "continuity", "location", "xyz"]
        # Each value: [B, N, feature_dim]

        parts = split_constraint_tensor(constraints)
        streams = {}

        for name in ["primitive_type", "direction", "dimension", "continuity", "location"]:
            component = parts[name]
            # [B, N, D_component] -> [B, N, 3 + D_component]
            concatenated = torch.cat([xyz, component], dim=-1)
            streams[name] = self.encoders[name](concatenated)

        # xyz-only stream (main constraint stream)
        streams["xyz"] = self.encoders["xyz"](xyz)

        return streams


class SharedNeighborhoodSampler(nn.Module):
    """
    Performs FPS and KNN once per hierarchy level, returns shared indices and relative coordinates.

    Used by all six feature streams to ensure they operate on the same spatial neighborhoods.
    """

    def __init__(self, n_center: int, n_near: int):
        super().__init__()
        self.n_center = n_center
        self.n_near = n_near

    def forward(
        self, xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # xyz: [B, N, 3]
        # Returns:
        #   center_idx: [B, S]
        #   group_idx: [B, S, K]
        #   center_xyz: [B, S, 3]
        #   relative_xyz: [B, S, K, 3]

        bsz, n_points, _ = xyz.shape
        n_center = min(self.n_center, n_points)
        n_near = min(self.n_near, n_points)

        # FPS to select center indices
        if n_center >= n_points:
            center_idx = torch.arange(n_points, device=xyz.device).view(1, n_points).repeat(bsz, 1)
        else:
            center_idx = utils.fps(xyz, n_center)

        # Get center coordinates
        center_xyz = utils.index_points(xyz, center_idx)  # [B, S, 3]

        # KNN: find k nearest neighbors to each center
        dists = torch.cdist(center_xyz, xyz)  # [B, S, N]
        group_idx = dists.topk(k=n_near, dim=-1, largest=False).indices  # [B, S, K]

        # Get relative coordinates
        grouped_xyz = utils.index_points(xyz, group_idx)  # [B, S, K, 3]
        relative_xyz = grouped_xyz - center_xyz.unsqueeze(2)  # [B, S, K, 3]

        return center_idx, group_idx, center_xyz, relative_xyz


class LocalVectorAttentionAggregator(nn.Module):
    """
    Aggregates neighbor features to center using local vector attention.

    Replaces max pooling with learned attention that explicitly uses feature differences.

    Output: [B, S, C_out]
    """

    def __init__(self, channel_in: int, channel_out: int):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out

        # Position encoding
        self.position_mlp = PointwiseMLP((3, channel_out, channel_out))

        # Query, Key, Value projections
        self.query_proj = nn.Linear(channel_in, channel_out)
        self.key_proj = nn.Linear(channel_in, channel_out)
        self.value_proj = nn.Linear(channel_in, channel_out)

        # Attention logits MLP
        self.attention_mlp = nn.Sequential(
            nn.Linear(channel_out, channel_out),
            nn.SiLU(inplace=True),
            nn.Linear(channel_out, channel_out),
        )

        # Center and output projections
        self.center_proj = nn.Linear(channel_in, channel_out)

        # Residual normalization and FFN
        self.norm1 = nn.LayerNorm(channel_out)
        self.norm2 = nn.LayerNorm(channel_out)
        self.ffn = nn.Sequential(
            nn.Linear(channel_out, 2 * channel_out),
            nn.SiLU(inplace=True),
            nn.Linear(2 * channel_out, channel_out),
        )

    def forward(
        self,
        center_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        relative_xyz: torch.Tensor,
    ) -> torch.Tensor:
        # center_features: [B, S, C_in]
        # neighbor_features: [B, S, K, C_in]
        # relative_xyz: [B, S, K, 3]
        # Output: [B, S, C_out]

        B, S, K, _ = neighbor_features.shape

        # Position encoding
        # relative_xyz [B, S, K, 3] -> position_encoded [B, S, K, C_out]
        flat_relative_xyz = relative_xyz.view(B * S, K, 3)
        flat_pos_enc = self.position_mlp(flat_relative_xyz)  # [B*S, K, C_out]
        pos_enc = flat_pos_enc.view(B, S, K, self.channel_out)  # [B, S, K, C_out]

        # Query from center, Key and Value from neighbors
        q = self.query_proj(center_features)  # [B, S, C_out]
        k = self.key_proj(neighbor_features)  # [B, S, K, C_out]
        v = self.value_proj(neighbor_features)  # [B, S, K, C_out]

        # Expand query for broadcasting
        q_expanded = q.unsqueeze(2)  # [B, S, 1, C_out]

        # Compute relation: key - query + position_encoding
        relation = k - q_expanded + pos_enc  # [B, S, K, C_out]

        # Attention logits and weights
        attention_logits = self.attention_mlp(relation)  # [B, S, K, C_out]
        attention = F.softmax(attention_logits, dim=2)  # [B, S, K, C_out], softmax over K

        # Aggregate: attention * (value + position_encoding)
        message = attention * (v + pos_enc)  # [B, S, K, C_out]
        aggregated = message.sum(dim=2)  # [B, S, C_out]

        # Residual connection and normalization
        center_proj = self.center_proj(center_features)  # [B, S, C_out]
        x = self.norm1(center_proj + aggregated)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x


class MultiStreamSetAbstractionLayer(nn.Module):
    """
    Applies shared neighborhood sampling and independent vector attention for each stream.

    Ensures all six streams use identical spatial grouping but independent aggregation.
    """

    def __init__(self, n_center: int, n_near: int, channel_in: int, channel_out: int):
        super().__init__()
        self.sampler = SharedNeighborhoodSampler(n_center, n_near)

        # Independent aggregators for each stream
        self.aggregators = nn.ModuleDict(
            {
                "primitive_type": LocalVectorAttentionAggregator(channel_in, channel_out),
                "direction": LocalVectorAttentionAggregator(channel_in, channel_out),
                "dimension": LocalVectorAttentionAggregator(channel_in, channel_out),
                "continuity": LocalVectorAttentionAggregator(channel_in, channel_out),
                "location": LocalVectorAttentionAggregator(channel_in, channel_out),
                "xyz": LocalVectorAttentionAggregator(channel_in, channel_out),
            }
        )

    def forward(self, xyz: torch.Tensor, streams: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # xyz: [B, N, 3]
        # streams: dict with 6 features [B, N, C_in]
        # Output: (center_xyz [B, S, 3], updated_streams dict [B, S, C_out])

        # Shared sampling: call once
        center_idx, group_idx, center_xyz, relative_xyz = self.sampler(xyz)

        # Gather center and neighbor features for each stream
        updated_streams = {}
        for stream_name, features in streams.items():
            # features: [B, N, C_in]
            center_features = utils.index_points(features, center_idx)  # [B, S, C_in]
            neighbor_features = utils.index_points(features, group_idx)  # [B, S, K, C_in]

            # Apply aggregator
            aggregated = self.aggregators[stream_name](center_features, neighbor_features, relative_xyz)
            updated_streams[stream_name] = aggregated  # [B, S, C_out]

        return center_xyz, updated_streams


class ComponentToConstraintCrossAttention(nn.Module):
    """
    Updates only the main constraint stream using the five component streams as context.

    Component streams remain unchanged (no feedback).
    Performs point-wise attention over the 5 components.
    """

    def __init__(self, channel_dim: int, n_heads: int = 4):
        super().__init__()
        self.channel_dim = channel_dim
        self.n_heads = n_heads

        # Multi-head cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=channel_dim,
            num_heads=n_heads,
            batch_first=True,
            dtype=torch.float32,
        )

        # Residual normalization and FFN
        self.norm1 = nn.LayerNorm(channel_dim)
        self.norm2 = nn.LayerNorm(channel_dim)
        self.ffn = nn.Sequential(
            nn.Linear(channel_dim, 2 * channel_dim),
            nn.SiLU(inplace=True),
            nn.Linear(2 * channel_dim, channel_dim),
        )

    def forward(
        self,
        constraint_feature: torch.Tensor,
        primitive_feature: torch.Tensor,
        direction_feature: torch.Tensor,
        dimension_feature: torch.Tensor,
        continuity_feature: torch.Tensor,
        location_feature: torch.Tensor,
    ) -> torch.Tensor:
        # constraint_feature: [B, S, C]
        # component_features: [B, S, C] each
        # Output: updated constraint_feature [B, S, C]

        B, S, C = constraint_feature.shape

        # Stack components into context [B, S, 5, C]
        component_context = torch.stack(
            [primitive_feature, direction_feature, dimension_feature, continuity_feature, location_feature],
            dim=2,
        )

        # Reshape for point-wise attention: treat each point independently
        # query: [B, S, 1, C] -> [B*S, 1, C]
        # context: [B, S, 5, C] -> [B*S, 5, C]
        query = constraint_feature.unsqueeze(2).reshape(B * S, 1, C)
        context = component_context.reshape(B * S, 5, C)

        # Apply multi-head attention
        updated, _ = self.attention(query, context, context)  # [B*S, 1, C]
        updated = updated.reshape(B, S, C)

        # Residual and normalization
        x = self.norm1(constraint_feature + updated)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x


class MultiStreamConstraintEncoder(nn.Module):
    """
    Hierarchical encoder applying multiple levels of set abstraction and cross-attention.

    Tracks all streams and xyz coordinates at each level for decoder skip connections.
    """

    def __init__(self, feature_dim: int = 96, latent_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        # Hierarchy levels
        self.sa1 = MultiStreamSetAbstractionLayer(512, 32, feature_dim, 160)
        self.ca1 = ComponentToConstraintCrossAttention(160)

        self.sa2 = MultiStreamSetAbstractionLayer(128, 32, 160, 320)
        self.ca2 = ComponentToConstraintCrossAttention(320)

    def forward(
        self, xyz: torch.Tensor, streams: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        # xyz: [B, N, 3]
        # streams: dict of [B, N, feature_dim]
        # Output: (final_streams dict, history dict with xyz and streams at each level)

        history = {
            "xyz": [xyz],
            "streams": [streams],
        }

        # Level 1: Set abstraction
        xyz1, streams1 = self.sa1(xyz, streams)
        history["xyz"].append(xyz1)
        history["streams"].append(streams1)

        # Level 1: Cross-attention (update constraint stream only)
        streams1["xyz"] = self.ca1(
            streams1["xyz"],
            streams1["primitive_type"],
            streams1["direction"],
            streams1["dimension"],
            streams1["continuity"],
            streams1["location"],
        )
        history["streams"][-1] = streams1

        # Level 2: Set abstraction
        xyz2, streams2 = self.sa2(xyz1, streams1)
        history["xyz"].append(xyz2)
        history["streams"].append(streams2)

        # Level 2: Cross-attention (update constraint stream only)
        streams2["xyz"] = self.ca2(
            streams2["xyz"],
            streams2["primitive_type"],
            streams2["direction"],
            streams2["dimension"],
            streams2["continuity"],
            streams2["location"],
        )
        history["streams"][-1] = streams2

        return streams2, history


# class ConstraintFeaturePropagation(nn.Module):
#     """
#     Propagates the final constraint stream back to original point resolution via KNN interpolation.

#     Used by diffusion decoder to obtain full-resolution constraint features.
#     """

#     def __init__(self, stage_channels: tuple[int, ...]):
#         super().__init__()

#         self.projs = nn.ModuleList()
#         for i in range(len(stage_channels) - 1):
#             self.projs.append(nn.Linear(stage_channels[i+1], stage_channels[i]))

#     def forward(
#         self,
#         constraint_feature: torch.Tensor,
#         constraint_skip_features: list,
#         xyz_hierarchy: list,
#         original_xyz: torch.Tensor,
#     ) -> torch.Tensor:
#         # constraint_feature: [B, S_L, C] at lowest resolution
#         # constraint_skip_features: list of [B, S_i, C] at each hierarchy level
#         # xyz_hierarchy: list of [B, S_i, 3] at each hierarchy level (reversed order for propagation)
#         # original_xyz: [B, N, 3] at original resolution
#         # Output: [B, N, C] full-resolution constraint features

#         B, _, C = constraint_feature.shape
#         device = constraint_feature.device

#         current_feature = constraint_feature  # [B, S_L, C]

#         # Propagate backwards through hierarchy
#         # Skip features are in reverse order (finest to coarsest), so we go from coarsest to finest
#         for i in range(len(constraint_skip_features) - 1, 0, -1):
#             current_xyz = xyz_hierarchy[i]  # [B, S_i, 3]
#             target_xyz = xyz_hierarchy[i - 1]  # [B, S_{i-1}, 3]
#             skip_feature = constraint_skip_features[i - 1]  # [B, S_{i-1}, C]

#             # KNN interpolation: min(3, num_source_points)
#             k = min(3, current_xyz.shape[1])

#             # Compute distances
#             dists = torch.cdist(target_xyz, current_xyz)  # [B, S_{i-1}, S_i]
#             knn_indices = dists.topk(k=k, dim=-1, largest=False).indices  # [B, S_{i-1}, k]
#             knn_dists = dists.topk(k=k, dim=-1, largest=False).values  # [B, S_{i-1}, k]

#             # Clamp distances to avoid division by zero
#             eps = 1e-8
#             knn_dists = torch.clamp(knn_dists, min=eps)

#             # Inverse distance weights
#             weights = 1.0 / knn_dists  # [B, S_{i-1}, k]
#             weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)  # normalize

#             # Gather features and interpolate
#             batch_idx = torch.arange(B, device=device).view(B, 1, 1)
#             knn_features = current_feature[batch_idx, knn_indices]  # [B, S_{i-1}, k, C]

#             # Weighted average
#             interpolated = (knn_features * weights.unsqueeze(-1)).sum(dim=2)  # [B, S_{i-1}, C]

#             # Fuse with skip feature
#             proj_interpolated = self.projs[i-1](interpolated)
#             current_feature = proj_interpolated + skip_feature  # [B, S_{i-1}, C]

#         # Final propagation to original resolution
#         if original_xyz.shape[1] != xyz_hierarchy[0].shape[1]:
#             current_xyz = xyz_hierarchy[0]  # [B, S_0, 3]
#             k = min(3, current_xyz.shape[1])

#             dists = torch.cdist(original_xyz, current_xyz)  # [B, N, S_0]
#             knn_indices = dists.topk(k=k, dim=-1, largest=False).indices  # [B, N, k]
#             knn_dists = dists.topk(k=k, dim=-1, largest=False).values  # [B, N, k]

#             eps = 1e-8
#             knn_dists = torch.clamp(knn_dists, min=eps)

#             weights = 1.0 / knn_dists
#             weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)

#             batch_idx = torch.arange(B, device=device).view(B, 1, 1)
#             knn_features = current_feature[batch_idx, knn_indices]  # [B, N, k, C]

#             current_feature = (knn_features * weights.unsqueeze(-1)).sum(dim=2)  # [B, N, C]

#         return current_feature


class ConstraintFeaturePropagation(nn.Module):
    """
    自底向上（从粗到精）传播特征。
    将低通道的浅层 skip 特征投影（升维）到当前高通道维度进行融合，确保全程通道数为 320。
    """
    def __init__(self, stage_channels: tuple[int, ...]):
        super().__init__()
        # stage_channels: (96, 160, 320)
        # 实例化投影层：
        # self.projs[0]: Linear(in_features=96, out_features=320)
        # self.projs[1]: Linear(in_features=160, out_features=320)
        self.projs = nn.ModuleList([
            nn.Linear(stage_channels[0], stage_channels[-1]), 
            nn.Linear(stage_channels[1], stage_channels[-1]), 
        ])

    def forward(
        self,
        constraint_feature: torch.Tensor,
        constraint_skip_features: list[torch.Tensor],
        xyz_hierarchy: list[torch.Tensor],
        original_xyz: torch.Tensor,
    ) -> torch.Tensor:
        # 输入维度解析:
        # constraint_feature:       [B, S_L2, 320]  (最底层粗糙特征, S_L2=128)
        # constraint_skip_features: [[B, N, 96], [B, S_L1, 160], [B, S_L2, 320]] (从精到粗)
        # xyz_hierarchy:            [[B, N, 3],  [B, S_L1, 3],   [B, S_L2, 3]]   (从精到粗)
        # original_xyz:             [B, N, 3]       (原始高分辨率点云坐标, N=1024)

        B, _, C = constraint_feature.shape  # B=BatchSize, C=320
        device = constraint_feature.device
        current_feature = constraint_feature  # 初始状态: [B, 128, 320]

        # 级联反向遍历：i 从 2 递减到 1
        for i in range(len(constraint_skip_features) - 1, 0, -1):
            current_xyz = xyz_hierarchy[i]      # 第一次循环(i=2): [B, 128, 3]  | 第二次(i=1): [B, 512, 3]
            target_xyz = xyz_hierarchy[i - 1]    # 第一次循环(i=2): [B, 512, 3]  | 第二次(i=1): [B, N, 3]
            skip_feature = constraint_skip_features[i - 1] # 第一次(i=2): [B, 512, 160] | 第二次(i=1): [B, N, 96]

            # KNN 插值核心逻辑
            k = min(3, current_xyz.shape[1])
            
            # 计算目标点到当前源点的距离矩阵
            dists = torch.cdist(target_xyz, current_xyz)  # 第一次: [B, 512, 128] | 第二次: [B, N, 512]
            
            # 获取最近的 K 个点的索引和距离
            topk_out = dists.topk(k=k, dim=-1, largest=False)
            knn_indices = topk_out.indices  # 第一次: [B, 512, k] | 第二次: [B, N, k]
            knn_dists = torch.clamp(topk_out.values, min=1e-8) # 第一次: [B, 512, k] | 第二次: [B, N, k]

            # 倒数距离权重计算与归一化
            weights = 1.0 / knn_dists  # 第一次: [B, 512, k] | 第二次: [B, N, k]
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8) # 第一次: [B, 512, k] | 第二次: [B, N, k]

            # 索引强提并做加权平均插值
            batch_idx = torch.arange(B, device=device).view(B, 1, 1) # [B, 1, 1]
            knn_features = current_feature[batch_idx, knn_indices]  # 第一次: [B, 512, k, 320] | 第二次: [B, N, k, 320]
            
            # 插值后特征
            interpolated = (knn_features * weights.unsqueeze(-1)).sum(dim=2) # 第一次: [B, 512, 320] | 第二次: [B, N, 320]

            # 核心修改：利用对应的 Linear 层将低维的 skip 特征（160 或 96）投影升维到 320 维
            proj_skip = self.projs[i - 1](skip_feature) # 第一次: [B, 512, 320] | 第二次: [B, N, 320]
            
            # 特征对齐相加
            current_feature = interpolated + proj_skip  # 第一次: [B, 512, 320] | 第二次: [B, N, 320]

        # 最终检查：若点数未对齐，强制最后一次 KNN 传播到原始分辨率
        if original_xyz.shape[1] != xyz_hierarchy[0].shape[1]:
            current_xyz = xyz_hierarchy[0]  # [B, S_0, 3]
            k = min(3, current_xyz.shape[1])

            dists = torch.cdist(original_xyz, current_xyz)  # [B, N, S_0]
            topk_out_final = dists.topk(k=k, dim=-1, largest=False)
            knn_indices = topk_out_final.indices  # [B, N, k]
            knn_dists = torch.clamp(topk_out_final.values, min=1e-8)  # [B, N, k]

            weights = 1.0 / knn_dists  # [B, N, k]
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # [B, N, k]

            batch_idx = torch.arange(B, device=device).view(B, 1, 1)  # [B, 1, 1]
            knn_features = current_feature[batch_idx, knn_indices]  # [B, N, k, 320]

            current_feature = (knn_features * weights.unsqueeze(-1)).sum(dim=2)  # [B, N, 320]

        return current_feature  # 最终输出: [B, N, 320]
    



class CstNetStage2Classifier(nn.Module):
    """
    Constraint-aware classification head using multi-stream encoder.

    Uses only the main constraint stream for classification.
    """

    def __init__(self, n_classes: int, feature_dim: int = 96, latent_dim: int = 512):
        super().__init__()
        self.constraint_init = ConstraintStreamInitializer(feature_dim)
        self.encoder = MultiStreamConstraintEncoder(feature_dim, latent_dim)
        self.proj = nn.Sequential(
            nn.Linear(320, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, xyz: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        if constraints.shape[-1] != CONSTRAINT_DIM:
            raise ValueError(f"expected constraints [B, N, {CONSTRAINT_DIM}], got {tuple(constraints.shape)}")

        # Initialize streams
        streams = self.constraint_init(xyz, constraints)

        # Encode through hierarchy
        final_streams, _ = self.encoder(xyz, streams)

        # Use only constraint stream (xyz key in dict)
        constraint_final = final_streams["xyz"]  # [B, S, C]

        # Global pooling
        pooled = constraint_final.max(dim=1)[0]  # [B, C]

        # Project to latent
        latent = self.proj(pooled)

        # Classification
        logits = self.head(latent)
        return F.log_softmax(logits, dim=-1)


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)


class CstNetStage2Diffusion(nn.Module):
    def __init__(self, feature_dim: int = 96, latent_dim: int = 512, time_dim: int = 128):
        super().__init__()
        self.constraint_init = ConstraintStreamInitializer(feature_dim)
        self.encoder = MultiStreamConstraintEncoder(feature_dim, latent_dim)
        
        # 传入完整的通道金字塔，确保内部投影层能正确构建
        self.propagation = ConstraintFeaturePropagation((feature_dim, 160, 320))
        self.time_embedding = SinusoidalTimestepEmbedding(time_dim)
        
        # 此时 3 + 320 + 512 + 128 = 963 能够完美对齐！
        self.decoder = PointwiseMLP(
            (3 + 320 + 320 + time_dim, 384, 192, 3),
            activate_last=False,
        )

    def forward(self, noisy_xyz: torch.Tensor, constraints: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if constraints.shape[-1] != CONSTRAINT_DIM:
            raise ValueError(f"expected constraints [B, N, {CONSTRAINT_DIM}], got {tuple(constraints.shape)}")

        # Initialize streams
        streams = self.constraint_init(noisy_xyz, constraints)

        # Encode through hierarchy (track history for propagation)
        final_streams, history = self.encoder(noisy_xyz, streams)

        # Get global latent from constraint stream
        constraint_final = final_streams["xyz"]  # [B, S, C]
        global_latent = constraint_final.max(dim=1)[0]  # [B, C]

        # Propagate constraint features back to original resolution
        constraint_skip_features = [s["xyz"] for s in history["streams"]]
        full_resolution_constraint = self.propagation(
            constraint_final,
            constraint_skip_features,
            history["xyz"],
            noisy_xyz,
        )  # [B, N, C]

        # Timestep embedding
        time_emb = self.time_embedding(timesteps)  # [B, time_dim]

        # Prepare decoder input
        B, N, _ = noisy_xyz.shape
        decoder_input = torch.cat(
            [
                noisy_xyz,
                full_resolution_constraint,
                global_latent.unsqueeze(1).expand(-1, N, -1),
                time_emb.unsqueeze(1).expand(-1, N, -1),
            ],
            dim=-1,
        )  # [B, N, 3 + 320 + latent_dim + time_dim]

        # Predict noise
        noise = self.decoder(decoder_input)
        return noise
