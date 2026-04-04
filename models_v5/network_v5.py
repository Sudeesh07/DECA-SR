from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ChannelPackDWT, LayerNorm2d, ResidualConvBlock, make_drop_path_schedule
from .decoder import ProgressiveReconstructionDecoder
from .deformable_extractor import DeformableLocalFeatureBlock
from .stereo_fusion import IterativeDisparityRefinement, StereoFusionBlock, resolve_default_stereo_ranges


@dataclass
class StereoSRV5Output:
    sr_left: torch.Tensor
    sr_right: torch.Tensor
    aux: dict[str, torch.Tensor]


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        num_heads: int,
        num_points: int,
        drop_paths: list[float],
        downsample: bool,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                DeformableLocalFeatureBlock(
                    channels=in_channels,
                    num_heads=num_heads,
                    num_points=num_points,
                    drop_path=drop_paths[i],
                )
                for i in range(depth)
            ]
        )
        # NEW: Haar DWT lossless downsampling instead of strided conv
        self.down = ChannelPackDWT(in_channels, out_channels) if downsample else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.blocks(x)
        if self.down is None:
            return skip, skip
        return skip, self.down(skip)


class StereoSRV5(nn.Module):
    """V5 stereo SR: V4 + Haar DWT downsampling + sigmoid stereo gating + iterative refinement + subband skip gates."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        stage_channels: tuple[int, int, int] = (64, 96, 128),
        enc_blocks: tuple[int, int, int] = (4, 6, 8),
        bottleneck_blocks: int = 4,
        decoder_blocks: tuple[int, int] = (4, 4),
        hr_blocks: tuple[int, int] = (3, 3),
        deform_heads: int = 4,
        deform_points: int = 9,
        stereo_heads: int = 4,
        stereo_points: int = 5,
        depa_max_disparity: int | None = None,
        corr_max_disp: int | None = None,
        sample_span: float = 2.0,
        residual_range: float = 1.5,
        drop_path_rate: float = 0.1,
        num_refine_iters: int = 2,
    ):
        super().__init__()
        if len(stage_channels) != 3 or len(enc_blocks) != 3:
            raise ValueError("stage_channels and enc_blocks must both have 3 entries")
        if scale not in (2, 4):
            raise ValueError(f"scale must be 2 or 4, got {scale}")

        if depa_max_disparity is None or corr_max_disp is None:
            default_depa, default_corr = resolve_default_stereo_ranges(scale)
            depa_max_disparity = default_depa if depa_max_disparity is None else depa_max_disparity
            corr_max_disp = default_corr if corr_max_disp is None else corr_max_disp

        self.scale = scale
        self.stage_channels = stage_channels
        self.depa_max_disparity = depa_max_disparity
        self.corr_max_disp = corr_max_disp
        self.feature_stride = 4

        c1, c2, c3 = stage_channels
        drop_schedules = make_drop_path_schedule(enc_blocks, drop_path_rate)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            LayerNorm2d(c1),
        )

        self.stage1 = EncoderStage(c1, c2, enc_blocks[0], deform_heads, deform_points, drop_schedules[0], True)
        self.stage2 = EncoderStage(c2, c3, enc_blocks[1], deform_heads, deform_points, drop_schedules[1], True)
        self.stage3 = EncoderStage(c3, c3, enc_blocks[2], deform_heads, deform_points, drop_schedules[2], False)

        deep_disp = max(1, depa_max_disparity // self.feature_stride)
        deep_corr = max(1, corr_max_disp // self.feature_stride)

        self.deep_fuse_l = StereoFusionBlock(
            channels=c3, max_disp=deep_disp, corr_max_disp=deep_corr,
            num_heads=stereo_heads, num_points=stereo_points,
            sample_span=sample_span, residual_range=residual_range,
        )
        self.deep_fuse_r = StereoFusionBlock(
            channels=c3, max_disp=deep_disp, corr_max_disp=deep_corr,
            num_heads=stereo_heads, num_points=stereo_points,
            sample_span=sample_span, residual_range=residual_range,
        )

        self.bottleneck = nn.Sequential(*[ResidualConvBlock(c3) for _ in range(bottleneck_blocks)])

        # NEW: Iterative disparity refinement
        self.disp_refine_l = IterativeDisparityRefinement(c3, num_iters=num_refine_iters)
        self.disp_refine_r = IterativeDisparityRefinement(c3, num_iters=num_refine_iters)

        self.bottleneck_fuse_l = StereoFusionBlock(
            channels=c3, max_disp=deep_disp, corr_max_disp=deep_corr,
            num_heads=stereo_heads, num_points=stereo_points,
            sample_span=sample_span, residual_range=residual_range,
        )
        self.bottleneck_fuse_r = StereoFusionBlock(
            channels=c3, max_disp=deep_disp, corr_max_disp=deep_corr,
            num_heads=stereo_heads, num_points=stereo_points,
            sample_span=sample_span, residual_range=residual_range,
        )

        self.decoder_left = ProgressiveReconstructionDecoder(
            stage_channels=stage_channels, scale=scale, out_channels=in_channels,
            decoder_blocks=decoder_blocks, hr_blocks=hr_blocks, hr_channels=c1,
        )
        self.decoder_right = ProgressiveReconstructionDecoder(
            stage_channels=stage_channels, scale=scale, out_channels=in_channels,
            decoder_blocks=decoder_blocks, hr_blocks=hr_blocks, hr_channels=c1,
        )

    def _encode_view(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1, x = self.stage1(x)
        s2, x = self.stage2(x)
        s3, _ = self.stage3(x)
        return s1, s2, s3

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> StereoSRV5Output:
        if left.shape != right.shape:
            raise ValueError(f"Left/right shapes must match, got {left.shape} vs {right.shape}")
        input_left = left
        input_right = right

        left = self.stem(left)
        right = self.stem(right)

        s1_l, s2_l, s3_l = self._encode_view(left)
        s1_r, s2_r, s3_r = self._encode_view(right)

        # Deep stereo fusion
        s3_l_ref = s3_l
        s3_r_ref = s3_r
        s3_l, deep_stats_l = self.deep_fuse_l(s3_l_ref, s3_r_ref, reverse=False)
        s3_r, deep_stats_r = self.deep_fuse_r(s3_r_ref, s3_l_ref, reverse=True)

        # Bottleneck processing
        b_l = self.bottleneck(s3_l)
        b_r = self.bottleneck(s3_r)

        # NEW: Iterative disparity refinement using deep fusion disparity as initialization
        refined_disp_l = self.disp_refine_l(deep_stats_l.disparity, b_l, b_r, reverse=False)
        refined_disp_r = self.disp_refine_r(deep_stats_r.disparity, b_r, b_l, reverse=True)

        # Bottleneck stereo fusion with refined disparity
        b_l_ref = b_l
        b_r_ref = b_r
        b_l, bottleneck_stats_l = self.bottleneck_fuse_l(b_l_ref, b_r_ref, reverse=False, disp_init=refined_disp_l)
        b_r, bottleneck_stats_r = self.bottleneck_fuse_r(b_r_ref, b_l_ref, reverse=True, disp_init=refined_disp_r)

        # Decode
        pred_l = self.decoder_left(b_l, s2_l, s1_l)
        pred_r = self.decoder_right(b_r, s2_r, s1_r)

        # Global residual
        up_l = F.interpolate(input_left, scale_factor=self.scale, mode="bicubic", align_corners=False)
        up_r = F.interpolate(input_right, scale_factor=self.scale, mode="bicubic", align_corners=False)

        sr_left = pred_l + up_l
        sr_right = pred_r + up_r

        aux = {
            "deep_disp_left": deep_stats_l.disparity,
            "deep_disp_right": deep_stats_r.disparity,
            "deep_conf_left": deep_stats_l.confidence,
            "deep_conf_right": deep_stats_r.confidence,
            "refined_disp_left": refined_disp_l,
            "refined_disp_right": refined_disp_r,
            "bottleneck_disp_left": bottleneck_stats_l.disparity,
            "bottleneck_disp_right": bottleneck_stats_r.disparity,
            "bottleneck_conf_left": bottleneck_stats_l.confidence,
            "bottleneck_conf_right": bottleneck_stats_r.confidence,
        }
        return StereoSRV5Output(sr_left=sr_left, sr_right=sr_right, aux=aux)


def build_model_v5(args: Any) -> StereoSRV5:
    return StereoSRV5(
        scale=getattr(args, "scale", 4),
        in_channels=getattr(args, "in_channels", 3),
        stage_channels=tuple(getattr(args, "stage_channels", (64, 96, 128))),
        enc_blocks=tuple(getattr(args, "enc_blocks", (4, 6, 8))),
        bottleneck_blocks=getattr(args, "bottleneck_blocks", 4),
        decoder_blocks=tuple(getattr(args, "decoder_blocks", (4, 4))),
        hr_blocks=tuple(getattr(args, "hr_blocks", (3, 3))),
        deform_heads=getattr(args, "deform_heads", 4),
        deform_points=getattr(args, "deform_points", 9),
        stereo_heads=getattr(args, "stereo_heads", 4),
        stereo_points=getattr(args, "stereo_points", 5),
        depa_max_disparity=getattr(args, "depa_max_disparity", None),
        corr_max_disp=getattr(args, "corr_max_disp", None),
        sample_span=getattr(args, "sample_span", 2.0),
        residual_range=getattr(args, "residual_range", 1.5),
        drop_path_rate=getattr(args, "stochastic_depth", 0.1),
        num_refine_iters=getattr(args, "num_refine_iters", 2),
    )
