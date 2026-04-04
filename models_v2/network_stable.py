"""
StereoSRNet Stable: separate network file using stable epipolar attention.

This file is intentionally separate from existing network files so old code paths
remain unchanged.
"""

from typing import List, Optional, Tuple
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.nafblock import LayerNorm2d
except ModuleNotFoundError:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from models.nafblock import LayerNorm2d
from models_v2.depa_stable import (
    StableBottleneck,
    StableDecoderStage,
    StableEncoderStage,
    SimplePixelShuffleHead,
)
from models_v2.frequency import LightweightFrequencyRefinement
from models_v2.occlusion import HybridOcclusionFusion


class StereoSRNetStable(nn.Module):
    """
    Stereo SR network that uses BidirectionalStableDEPA in encoder stages.

    Design keeps:
    - Stage-wise disparity scaling:
      stage_disp = depa_max_disparity // (2 ** i)
    - Bottleneck correlation prior + DEPA
    - Occlusion fusion
    - Simple PixelShuffle reconstruction head
    - Global residual learning
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        embed_dim: int = 48,
        enc_blocks: Optional[List[int]] = None,
        dec_blocks: Optional[List[int]] = None,
        depa_k: int = 8,
        depa_max_disparity: int = 32,
        num_heads: int = 4,
        corr_max_disp: int = 12,
        drop_path_rate: float = 0.1,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
    ):
        super().__init__()

        if enc_blocks is None:
            enc_blocks = [4, 6, 6]
        if dec_blocks is None:
            dec_blocks = [4, 4]

        self.scale = scale
        num_enc = len(enc_blocks)
        num_down = num_enc - 1

        if len(dec_blocks) != num_down:
            raise ValueError(
                f"dec_blocks length ({len(dec_blocks)}) must equal number of downsamples ({num_down})"
            )

        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )

        total_enc = sum(enc_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]

        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i, nb in enumerate(enc_blocks):
            rates = dpr[block_idx : block_idx + nb]
            block_idx += nb

            stage_disp = depa_max_disparity // (2 ** i)
            self.encoder_stages.append(
                StableEncoderStage(
                    channels=embed_dim,
                    num_blocks=nb,
                    depa_k=depa_k,
                    depa_max_disparity=stage_disp,
                    num_heads=num_heads,
                    downsample=(i < num_down),
                    drop_path_rates=rates,
                    sample_span=sample_span,
                    residual_range=residual_range,
                )
            )

        bottleneck_disp = depa_max_disparity // (2 ** num_down)
        self.bottleneck = StableBottleneck(
            channels=embed_dim,
            depa_k=depa_k,
            num_heads=num_heads,
            max_disparity=bottleneck_disp,
            corr_max_disp=corr_max_disp,
            sample_span=sample_span,
            residual_range=residual_range,
        )

        self.decoder_stages = nn.ModuleList(
            [StableDecoderStage(embed_dim, nb) for nb in dec_blocks]
        )

        self.freq_refine = LightweightFrequencyRefinement(embed_dim)
        self.occlusion = HybridOcclusionFusion(embed_dim)
        self.upsampler = SimplePixelShuffleHead(embed_dim, scale, out_channels=in_channels)

    def _encode(
        self, fl: torch.Tensor, fr: torch.Tensor
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor],
        torch.Tensor, torch.Tensor,
    ]:
        skips_l: List[torch.Tensor] = []
        skips_r: List[torch.Tensor] = []
        pre_l = fl
        pre_r = fr

        for i, stage in enumerate(self.encoder_stages):
            fl_skip, fr_skip, fl_next, fr_next, fl_pre, fr_pre = stage(fl, fr)
            skips_l.append(fl_skip)
            skips_r.append(fr_skip)
            if i == len(self.encoder_stages) - 1:
                pre_l, pre_r = fl_pre, fr_pre
            fl, fr = fl_next, fr_next

        return skips_l, skips_r, pre_l, pre_r

    def _decode(
        self,
        pre_target: torch.Tensor,
        pre_ref: torch.Tensor,
        skips_target: List[torch.Tensor],
    ) -> torch.Tensor:
        x = self.bottleneck(pre_target, pre_ref)

        for i, dec in enumerate(self.decoder_stages):
            skip_idx = len(self.encoder_stages) - 2 - i
            x = dec(x, skips_target[skip_idx])

        return self.freq_refine(x)

    def forward(
        self, img_left: torch.Tensor, img_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fl = self.shallow(img_left)
        fr = self.shallow(img_right)

        skips_l, skips_r, pre_l, pre_r = self._encode(fl, fr)

        xl = self._decode(pre_l, pre_r, skips_l)
        xr = self._decode(pre_r, pre_l, skips_r)

        fused_l, conf_l = self.occlusion(xl, xr)
        fused_r, conf_r = self.occlusion(xr, xl)

        sr_left = self.upsampler(fused_l)
        sr_right = self.upsampler(fused_r)

        lr_up_l = F.interpolate(img_left, scale_factor=self.scale, mode="bilinear", align_corners=False)
        lr_up_r = F.interpolate(img_right, scale_factor=self.scale, mode="bilinear", align_corners=False)
        sr_left = sr_left + lr_up_l
        sr_right = sr_right + lr_up_r

        conf_l = F.interpolate(conf_l, size=sr_left.shape[-2:], mode="bilinear", align_corners=False)
        conf_r = F.interpolate(conf_r, size=sr_right.shape[-2:], mode="bilinear", align_corners=False)

        return sr_left, sr_right, conf_l, conf_r


def stereo_sr_stable_tiny(scale: int = 4, **kw) -> StereoSRNetStable:
    return StereoSRNetStable(
        scale=scale,
        embed_dim=32,
        enc_blocks=[2, 4, 4],
        dec_blocks=[2, 2],
        depa_k=4,
        num_heads=2,
        corr_max_disp=8,
        **kw,
    )


def stereo_sr_stable_base(scale: int = 4, **kw) -> StereoSRNetStable:
    return StereoSRNetStable(
        scale=scale,
        embed_dim=48,
        enc_blocks=[4, 6, 6],
        dec_blocks=[4, 4],
        depa_k=8,
        num_heads=4,
        corr_max_disp=12,
        **kw,
    )


def stereo_sr_stable_large(scale: int = 4, **kw) -> StereoSRNetStable:
    return StereoSRNetStable(
        scale=scale,
        embed_dim=64,
        enc_blocks=[6, 8, 8, 8],
        dec_blocks=[4, 4, 4],
        depa_k=8,
        num_heads=4,
        corr_max_disp=16,
        drop_path_rate=0.2,
        **kw,
    )


def build_model_stable(args) -> StereoSRNetStable:
    """Build stable model from argparse-style namespace."""
    return StereoSRNetStable(
        scale=getattr(args, "scale", 4),
        in_channels=getattr(args, "in_channels", 3),
        embed_dim=getattr(args, "embed_dim", 48),
        enc_blocks=getattr(args, "enc_blocks", [4, 6, 6]),
        dec_blocks=getattr(args, "dec_blocks", [4, 4]),
        depa_k=getattr(args, "depa_k", 8),
        depa_max_disparity=getattr(args, "depa_max_disparity", 32),
        num_heads=getattr(args, "num_heads", 4),
        corr_max_disp=getattr(args, "corr_max_disp", 12),
        drop_path_rate=getattr(args, "stochastic_depth", 0.1),
        sample_span=getattr(args, "sample_span", 4.0),
        residual_range=getattr(args, "residual_range", 3.0),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = stereo_sr_stable_base(scale=4).to(device)

    left = torch.randn(1, 3, 30, 90, device=device)
    right = torch.randn(1, 3, 30, 90, device=device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")

    with torch.no_grad():
        sr_l, sr_r, c_l, c_r = model(left, right)

    print(f"Input:  {tuple(left.shape)}")
    print(f"SR_L:   {tuple(sr_l.shape)}")
    print(f"SR_R:   {tuple(sr_r.shape)}")
    print(f"Conf_L: {tuple(c_l.shape)}")
    print(f"Conf_R: {tuple(c_r.shape)}")
