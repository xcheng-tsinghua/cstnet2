from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from networks.feature_propagation import FeaturePropagation
from networks.stage2 import Stage2ConstraintBackbone


class Stage2SegmentationModel(nn.Module):
    """Constraint-aware MFCAD++ point-wise machining feature segmenter."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 64,
        encoder_channels: Sequence[int] = (128, 256, 512),
        decoder_channels: Sequence[int] = (256, 192, 128),
        global_context_dim: int = 128,
        norm_type: str = "ln",
        n_centers: Sequence[int] = (512, 128, 32),
        n_neighbors: Sequence[int] = (32, 32, 32),
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be read from metadata and be greater than one")
        if len(encoder_channels) != 3 or len(decoder_channels) != 3:
            raise ValueError("segmentation encoder and decoder must each contain three levels")

        self.num_classes = int(num_classes)
        encoder_channels = tuple(int(channel) for channel in encoder_channels)
        decoder_channels = tuple(int(channel) for channel in decoder_channels)

        self.backbone = Stage2ConstraintBackbone(
            feature_dim=feature_dim,
            stage_channels=encoder_channels,
            n_centers=n_centers,
            n_neighbors=n_neighbors,
            norm_type=norm_type,
        )

        # main_32 -> main_128 -> main_512 -> main_input
        self.fp3 = FeaturePropagation(
            encoder_channels[2] + encoder_channels[1], decoder_channels[0]
        )
        self.fp2 = FeaturePropagation(
            decoder_channels[0] + encoder_channels[0], decoder_channels[1]
        )
        self.fp1 = FeaturePropagation(
            decoder_channels[1] + feature_dim, decoder_channels[2]
        )

        self.global_projection = nn.Sequential(
            nn.Linear(2 * encoder_channels[2], global_context_dim, bias=False),
            nn.LayerNorm(global_context_dim),
            nn.GELU(),
        )
        final_dim = decoder_channels[2] + global_context_dim
        self.seg_head = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_classes),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        constraints: torch.Tensor,
        constraint_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pyramid = self.backbone(
            xyz=xyz,
            constraints=constraints,
            component_masks=constraint_masks,
            return_pyramid=True,
        )
        xyz_levels = pyramid["xyz"]
        main_levels = pyramid["main"]
        if len(xyz_levels) != 4 or len(main_levels) != 4:
            raise RuntimeError("segmentation backbone must return four pyramid resolutions")

        decoded_128 = self.fp3(
            xyz_fine=xyz_levels[2],
            xyz_coarse=xyz_levels[3],
            feat_fine=main_levels[2],
            feat_coarse=main_levels[3],
        )
        decoded_512 = self.fp2(
            xyz_fine=xyz_levels[1],
            xyz_coarse=xyz_levels[2],
            feat_fine=main_levels[1],
            feat_coarse=decoded_128,
        )
        decoded_full = self.fp1(
            xyz_fine=xyz_levels[0],
            xyz_coarse=xyz_levels[1],
            feat_fine=main_levels[0],
            feat_coarse=decoded_512,
        )

        deepest_main = main_levels[3]
        global_feature = torch.cat(
            [deepest_main.max(dim=1).values, deepest_main.mean(dim=1)],
            dim=-1,
        )
        global_context = self.global_projection(global_feature)
        global_context = global_context.unsqueeze(1).expand(-1, xyz.shape[1], -1)
        return self.seg_head(torch.cat([decoded_full, global_context], dim=-1))

