"""DSRPSNRNet: targeted stereo fusion + stronger progressive decoder."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nafblock import LayerNorm2d
from models_v3.blocks import DLFEBlock, HybridStereoFusion
from models_v3.decoder import ProgressiveDecoder


class DSRPSNRNet(nn.Module):
    """
    Stereo SR network focused on PSNR improvements:
    - stronger single-view local feature extraction
    - targeted stereo fusion only at deep stage and optional bottleneck
    - stronger progressive reconstruction (x2 -> x4) with concat skip fusion
    - no frequency refinement and no occlusion fusion
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        embed_dim: int = 64,
        enc_blocks: Sequence[int] = (4, 6, 6),
        depa_k: int = 8,
        depa_max_disparity: int | None = None,
        corr_max_disp: int | None = None,
        use_bottleneck_stereo: bool = True,
        num_heads: int = 4,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
    ):
        super().__init__()
        if len(enc_blocks) != 3:
            raise ValueError(f"enc_blocks must have 3 stages, got {enc_blocks}")
        if scale not in (2, 4):
            raise ValueError(f"scale must be 2 or 4, got {scale}")

        if depa_max_disparity is None:
            depa_max_disparity = 64 if scale == 2 else 32
        if corr_max_disp is None:
            corr_max_disp = 16 if scale == 2 else 12

        # corr_max_disp is kept for config compatibility/ablation logging.
        self.corr_max_disp = corr_max_disp
        self.scale = scale
        self.use_bottleneck_stereo = use_bottleneck_stereo

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )

        self.stage1 = nn.Sequential(*[DLFEBlock(embed_dim) for _ in range(enc_blocks[0])])
        self.down1 = nn.Sequential(
            LayerNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
        )

        self.stage2 = nn.Sequential(*[DLFEBlock(embed_dim) for _ in range(enc_blocks[1])])
        self.down2 = nn.Sequential(
            LayerNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
        )

        self.stage3 = nn.Sequential(*[DLFEBlock(embed_dim) for _ in range(enc_blocks[2])])

        deep_disp = max(1, depa_max_disparity // 4)
        bottleneck_disp = max(1, depa_max_disparity // 4)

        self.deep_fuse_l = HybridStereoFusion(
            channels=embed_dim,
            depa_k=depa_k,
            num_heads=num_heads,
            max_disparity=deep_disp,
            sample_span=sample_span,
            residual_range=residual_range,
        )
        self.deep_fuse_r = HybridStereoFusion(
            channels=embed_dim,
            depa_k=depa_k,
            num_heads=num_heads,
            max_disparity=deep_disp,
            sample_span=sample_span,
            residual_range=residual_range,
        )

        self.bottleneck_refine = nn.Sequential(
            DLFEBlock(embed_dim),
            DLFEBlock(embed_dim),
        )

        self.bottleneck_fuse_l = HybridStereoFusion(
            channels=embed_dim,
            depa_k=depa_k,
            num_heads=num_heads,
            max_disparity=bottleneck_disp,
            sample_span=sample_span,
            residual_range=residual_range,
        )
        self.bottleneck_fuse_r = HybridStereoFusion(
            channels=embed_dim,
            depa_k=depa_k,
            num_heads=num_heads,
            max_disparity=bottleneck_disp,
            sample_span=sample_span,
            residual_range=residual_range,
        )

        self.decoder_left = ProgressiveDecoder(embed_dim, out_channels=in_channels, scale=scale)
        self.decoder_right = ProgressiveDecoder(embed_dim, out_channels=in_channels, scale=scale)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.stage1(x)
        x = self.down1(s1)
        s2 = self.stage2(x)
        x = self.down2(s2)
        s3 = self.stage3(x)
        return s1, s2, s3

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fl = self.stem(left)
        fr = self.stem(right)

        s1_l, s2_l, s3_l = self._encode(fl)
        s1_r, s2_r, s3_r = self._encode(fr)

        # Targeted stereo fusion at deep stage only.
        s3_l_ref = s3_l
        s3_r_ref = s3_r
        s3_l = self.deep_fuse_l(s3_l_ref, s3_r_ref)
        s3_r = self.deep_fuse_r(s3_r_ref, s3_l_ref)

        b_l = self.bottleneck_refine(s3_l)
        b_r = self.bottleneck_refine(s3_r)

        if self.use_bottleneck_stereo:
            b_l_ref = b_l
            b_r_ref = b_r
            b_l = self.bottleneck_fuse_l(b_l_ref, b_r_ref)
            b_r = self.bottleneck_fuse_r(b_r_ref, b_l_ref)

        pred_l = self.decoder_left(b_l, s2_l, s1_l)
        pred_r = self.decoder_right(b_r, s2_r, s1_r)

        up_l = F.interpolate(left, scale_factor=self.scale, mode="bilinear", align_corners=False)
        up_r = F.interpolate(right, scale_factor=self.scale, mode="bilinear", align_corners=False)

        sr_left = pred_l + up_l
        sr_right = pred_r + up_r
        return sr_left, sr_right


def build_model_v3(scale: int = 4, **kwargs) -> DSRPSNRNet:
    """Convenience builder for DSRPSNRNet."""
    return DSRPSNRNet(scale=scale, **kwargs)
